import collections
from functools import partial as bind

import elements
import embodied
import numpy as np
from tqdm import tqdm

from embodied.run.video_utils import VideoRecorder

min_report = 64


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  last_score = {'score': '-', 'length': '-'}
  last_loss = '-'

  # Video recording
  video_recorder = VideoRecorder(
      logdir, video_every=getattr(args, 'video_every', 5000))

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      ep_score = result.pop('score')
      ep_length = result.pop('length')
      logger.add({
          'score': ep_score,
          'length': ep_length,
      }, prefix='episode')
      if np.isfinite(ep_score):
        last_score['score'] = f"{ep_score:.2f}"
      if np.isfinite(ep_length):
        last_score['length'] = f"{ep_length:.0f}"
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(lambda tran, worker: replay.add(tran, worker))
  driver.on_step(logfn)

  # Video recording callback
  def collect_video_frame(tran, worker):
    if worker != 0:
      return
    env = driver.envs[0] if hasattr(driver, 'envs') and driver.envs else None
    video_recorder.on_step(int(step), env, tran)

  driver.on_step(collect_video_frame)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    nonlocal last_loss
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
      # Extract total loss for progress bar display
      loss_keys = [k for k in mets if k.startswith('loss/')]
      if loss_keys:
        total_loss = sum(float(mets[k]) for k in loss_keys if np.isfinite(mets[k]))
        if np.isfinite(total_loss):
          last_loss = f"{total_loss:.1f}"

  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)

  pbar = tqdm(total=args.steps, dynamic_ncols=True, desc='Train', initial=int(step))
  while step < args.steps:

    driver(policy, steps=10)

    # Only run report when enough data is available for a full report batch.
    if should_report(step) and len(replay) >= min_report:
      if stream_report is None:
        stream_report = iter(agent.stream(make_stream(replay, 'report')))
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

    # Update progress visualization.
    pbar.n = int(step)
    pbar.set_postfix({
        'score': last_score.get('score', '-'),
        'len': last_score.get('length', '-'),
        'loss': last_loss if last_loss != '-' else '-',
        'fps': f"{policy_fps.result():.1f}",
    })
    pbar.refresh()

  logger.close()
  pbar.close()
