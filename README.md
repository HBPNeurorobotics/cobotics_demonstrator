# Installation instructions

    # Clone experiment
	cd ~/.opt/nrpStorage
    git clone https://github.com/HBPNeurorobotics/cobotics_demonstrator.git
    cd cobotics_demonstrator
    # Copy Model files
    cp -r Models/* $HBP/Models
    # Copy ROS packages (contains Kuka IIWA MoveIt configuration)
    cp -r GazeboRosPkgs/src/* $HBP/GazeboRosPkgs/src/
    # Link new models
    cd $HBP/Models
    ./create-symlinks.sh
    # Compile ROS Packages
    cd $HBP/GazeboRosPkgs
    catkin_make -j8
