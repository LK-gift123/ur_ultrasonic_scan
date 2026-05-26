#!/bin/bash

ros2 control switch_controllers \
--activate ur7e_manipulator_controller \
--deactivate forward_velocity_controller