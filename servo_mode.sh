#!/bin/bash

ros2 control switch_controllers \
--activate forward_velocity_controller \
--deactivate ur7e_manipulator_controller
