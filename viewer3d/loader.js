/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import { LoaderUtils } from "three";
import { XacroLoader } from "xacro-parser";
import URDFLoader from "urdf-loader";

import Franka from "./robots/franka.js";


function loadRobotModel(robot) {
    return new Promise((resolve, reject) => {
        const xacroLoader = new XacroLoader();
        xacroLoader.inOrder = true;
        xacroLoader.requirePrefix = true;
        xacroLoader.localProperties = true;

        xacroLoader.rospackCommands.find = (...args) => {
            return robot.root + args[0];
        }

        xacroLoader.load(
            robot.xacro,
            (xml) => {
                const urdfLoader = new URDFLoader();
                urdfLoader.packages = robot.packages;
                urdfLoader.workingPath = LoaderUtils.extractUrlBase(robot.xacro);

                urdfLoader.loadMeshCb = (path, manager, done) => {
                    urdfLoader.defaultMeshLoader(path, manager, (mesh) => {
                        enableShadows(mesh);
                        done(mesh);
                    });
                };

                let model = urdfLoader.parse(xml);
                resolve(model);
            },
            (error) => {
                console.error(error);
                reject(error);
            }
        );
    });
}


function enableShadows(object) {
    if (object.isMesh) {
        object.castShadow = true;
        object.receiveShadow = true;
    } else if (object.isGroup) {
        object.children.forEach(child => { enableShadows(child) });
    }
}


export function loadRobot() {
    let robot = new Franka();

    return loadRobotModel(robot)
        .then(model => {
            robot.init(model);
            robot.setPose(robot.defaultPose);
            return Promise.resolve(robot);
        }, reason => {
            console.error(reason);
            return Promise.reject();
        });
}
