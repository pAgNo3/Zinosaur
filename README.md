**Problem Statement**

The aim here is to make a multiterrain capable and robust wheeled biped robot which can demonstrate excellent agility and stability. The robot can be used for multiple applications in forest-like and mountainous areas for surveillance, searching, exploration, etc. 

<img width="425" height="567" alt="image" src="https://github.com/user-attachments/assets/f98ff854-6495-4d7b-a2a0-61d19f7a1ce3" />

<img width="760" height="531" alt="image" src="https://github.com/user-attachments/assets/941a1fd5-9e26-4d43-b2ff-011cb0ac6c15" />


**Current Progress**

#15/09/26: The bot is able to stand in mujoco using wheels only and is able to withstand kicks upto 70N, with good recovery. Back to back kicks are still a problem.
#16/09/26: The bot is able to stand In mujoco using its legs (not only wheels). Trained using SAC for 5L timesteps, entcoeff=15. ROS2 integration is still clanky so mujoco only.
