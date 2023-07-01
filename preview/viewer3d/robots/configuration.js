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

export default class RobotConfiguration {
    constructor() {
        // Root link of the robot
        this.robotRoot = null;

        // Root link of the tool of the robot (can be null if no tool)
        this.toolRoot = null;

        // Site to use as the TCP
        this.tcpSite = null;

        // Default pose of the robot
        this.defaultPose = {
        },

        this.jointPositionHelpers = {
            // Offsets to apply to the joint position helpers
            offsets: {},

            // Joint position helpers that must be inverted
            inverted: [],
        };
    }
}
