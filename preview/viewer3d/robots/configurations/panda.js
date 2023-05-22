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

import RobotConfiguration from '../configuration.js'


export default class PandaConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "link0";
        this.toolRoot = "hand";
        this.tcpSite = "tcp";

        this.defaultPose = {
            joint1: 0.5,
            joint2: -0.3,
            joint4: -1.8,
            joint6: 1.5,
            joint7: 1.0,
        };

        this.jointPositionHelpers.offsets = {
            joint1: -0.19,
            joint3: -0.12,
            joint5: -0.26,
            joint6: -0.015,
            joint7: 0.05,
        };

        this.jointPositionHelpers.inverted = [
            'joint4',
            'joint5',
            'joint6',
        ];
    }
}
