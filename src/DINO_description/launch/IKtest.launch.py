import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

def generate_launch_description():
    # Absolute paths to your scripts
    sim_script = os.path.abspath("simlauncher.py")
    IKtest_script = os.path.abspath("IKtester.py") # Your model.learn() script
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Set to "true" for headless training, "false" to open 3D window'
    )

    # 1. Start MuJoCo Sim Launcher Node (publishes /joint_states, /imu; listens to /robot_cmd)
    sim_process = ExecuteProcess(
        cmd=['python3', sim_script, '--headless', LaunchConfiguration('headless')],
        output='screen'
    )

    # 2. Start RL Training Script Process
    IKtest_process = ExecuteProcess(
        cmd=['python3', IKtest_script],
        output='screen'
    )

    # 3. Shutdown Handler: Automatically kill the simulation when training finishes
    shutdown_on_train_exit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=IKtest_process,
            on_exit=[
                EmitEvent(event=Shutdown(reason='heeee heee'))
            ]
        )
    )

    return LaunchDescription([
        headless_arg,
        sim_process,
        IKtest_process,
        shutdown_on_train_exit
    ])
