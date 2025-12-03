#!/bin/bash
# Launch the BlocksV2 simulator in HEADLESS mode
PACKAGED_ROOT="/ntfs-gen4-1tb/RoboticsProject/Colosseum/Packaged/BlocksV2"

PACKAGED_SIM="$PACKAGED_ROOT/Linux/BlocksV2.sh"
if [ -f "$PACKAGED_SIM" ]; then
    SIM_SCRIPT="$PACKAGED_SIM"
    echo "Using packaged build at $SIM_SCRIPT"
else
    echo "Simulator script not found at $PACKAGED_SIM"
    echo "Please run launch_blocks_sim.sh first to build the project."
    exit 1
fi

echo "Launching BlocksV2 Simulator (Headless)..."
# Performance optimizations:
# -RenderOffscreen: No window (but GPU rendering for cameras)
# Reduced render resolution but camera capture at desired resolution
# Disabled expensive features: motion blur, bloom, lens flares, eye adaptation
# Lower shadow quality, reduced view distance
# r.MaxFPS limits framerate to save GPU when RL doesn't need high FPS
export UE4_BYPASS_AUDIO=1

# Optimized exec commands for RL training
EXEC_CMDS="sg.ViewDistanceQuality=1;sg.AntiAliasingQuality=1;sg.ShadowQuality=1;sg.PostProcessQuality=1;sg.TextureQuality=2;sg.EffectsQuality=1;sg.FoliageQuality=1;sg.ShadingQuality=1;r.PostProcessAAQuality=1;r.TemporalAA.Algorithm=0;r.ScreenPercentage=100;r.ViewDistanceScale=0.6;r.DefaultFeature.AutoExposure=0;r.EyeAdaptationQuality=0;r.TonemapperGamma=2.2;r.SceneColorFringeQuality=0;r.DefaultFeature.MotionBlur=0;r.MotionBlurQuality=0;r.DefaultFeature.Bloom=0;r.DefaultFeature.LensFlare=0;r.LightShaftQuality=0;r.RefractionQuality=0;r.SSR.Quality=0;r.DepthOfFieldQuality=0;r.FastBlurThreshold=0;r.Tonemapper.Quality=0;r.MaxFPS=60"

"$SIM_SCRIPT" \
  -RenderOffscreen \
  -windowed -ResX=4 -ResY=4 -NoVSync -unattended -nosound -NoSplash \
  -ExecCmds="$EXEC_CMDS"
