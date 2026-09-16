import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.actions import EmitEvent
from launch.events import Shutdown

def generate_launch_description():
    # Absolute paths to your scripts
    sim_script = os.path.abspath("simlauncher.py")
    visual_script = os.path.abspath("zino_vis.py") 

    # 1. Start MuJoCo Sim Launcher Node (publishes /joint_states, /imu; listens to /robot_cmd)
    sim_process = ExecuteProcess(
        cmd=['python3', sim_script],
        output='screen'
    )

    # 2. Start RL Training Script Process
    visual_process = ExecuteProcess(
        cmd=['python3', visual_script],
        output='screen'
    )

    # 3. Shutdown Handler: Automatically kill the simulation when training finishes
    shutdown_on_train_exit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=visual_process,
            on_exit=[
                EmitEvent(event=Shutdown(reason='SAC Training Completed!'))
            ]
        )
    )

    return LaunchDescription([
        sim_process,
        visual_process,
        shutdown_on_train_exit
    ])
