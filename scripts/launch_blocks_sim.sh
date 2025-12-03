#!/bin/bash
# Launch the BlocksV2 simulator
PACKAGED_ROOT="/ntfs-gen4-1tb/RoboticsProject/Colosseum/Packaged/BlocksV2"
PROJECT_ROOT="/ntfs-gen4-1tb/RoboticsProject/Colosseum/Unreal/Environments/BlocksV2"
UE_ROOT="/ntfs-gen4-1tb/RoboticsProject/UnrealEngine"

# Check for packaged build firs
PACKAGED_SIM="$PACKAGED_ROOT/Linux/BlocksV2.sh"
if [ -f "$PACKAGED_SIM" ]; then
    SIM_SCRIPT="$PACKAGED_SIM"
    echo "Using packaged build at $SIM_SCRIPT"
else
    echo "Packaged simulator not found at $PACKAGED_SIM"
    echo "Building BlocksV2..."

    # Build command
    "$UE_ROOT/Engine/Build/BatchFiles/RunUAT.sh" BuildCookRun \
        -project="$PROJECT_ROOT/BlocksV2.uproject" \
        -noP4 -platform=Linux -clientconfig=Shipping -serverconfig=Shipping \
        -cook -build -stage -pak -archive \
        -archivedirectory="$PACKAGED_ROOT"

    # Check if build succeeded
    if [ $? -eq 0 ]; then
        echo "Build successful."
        if [ -f "$PACKAGED_SIM" ]; then
            SIM_SCRIPT="$PACKAGED_SIM"
            echo "Using new build at $SIM_SCRIPT"
        else
            echo "Error: Build succeeded but binary not found at $PACKAGED_SIM"
            exit 1
        fi
    else
        echo "Error: Build failed."
        exit 1
    fi
fi

echo "Launching BlocksV2 Simulator..."
"$SIM_SCRIPT" -windowed -ResX=1280 -ResY=720
