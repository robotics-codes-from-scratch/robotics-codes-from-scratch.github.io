import {
	LoaderUtils
} from "three";


import { XacroLoader } from "xacro-parser";
import URDFLoader from "urdf-loader";

import Franka from "./robots/franka.js";


let robot = new Franka();


function loadRobotModel(url) {
	return new Promise((resolve, reject) => {
		const xacroLoader = new XacroLoader();
		xacroLoader.inOrder = true;
		xacroLoader.requirePrefix = true;
		xacroLoader.localProperties = true;

		xacroLoader.rospackCommands.find = (...args) => {
            return robot.root + args[0];
		}

		xacroLoader.load(
			url,
			(xml) => {
                const urdfLoader = new URDFLoader();
                urdfLoader.packages = robot.packages;
                urdfLoader.workingPath = LoaderUtils.extractUrlBase(url);

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


export function loadRobot() {
    return loadRobotModel(robot.xacro)
        .then(model => {
            robot.init(model);
            robot.setPose(robot.defaultPose);
            return Promise.resolve(robot);
        }, reason => {
            console.error(reason);
            return Promise.reject();
        });
}
