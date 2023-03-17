/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileCopyrightText: Copyright © 2022 Nikolas Dahn
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 * SPDX-FileContributor: Nikolas Dahn
 *
 * SPDX-License-Identifier: MIT
 *
 * This file is a modification of the one implemented in https://github.com/ndahn/Rocksi
 *
 */

import Robot from './robotbase.js'


export default class Franka extends Robot {
    constructor() {
        super("Franka", "franka_description", "robots/panda_arm_hand.urdf.xacro");

        this.robotRoot = "panda_link0";
        this.handRoot = "panda_hand";

        this.defaultPose = {
            panda_joint1: 0.5,
            panda_joint2: -0.3,
            panda_joint4: -1.8,
            panda_joint6: 1.5,
            panda_joint7: 1.0,
        };

        this.tcp.parent = "panda_hand";
        this.tcp.position = [0, 0, 0.103394];

        this.jointPositionHelperOffsets = {
            panda_joint1: -0.19,
            panda_joint3: -0.12,
            panda_joint5: -0.26,
            panda_joint6: -0.015,
            panda_joint7: 0.05,
        };

        this.jointPositionHelperInverted = [
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
        ];
    }
}
