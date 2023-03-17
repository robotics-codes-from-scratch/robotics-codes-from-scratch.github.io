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

import { Object3D, Vector3, Quaternion, Euler, Bone } from 'three';
import { TWEEN } from 'three/examples/jsm/libs/tween.module.min.js';
import { traverse, getURL } from '../utils.js';
import JointPositionHelper from '../helpers/jointpositionhelper.js';


export const MODELS_ROOT = getURL("models/");
export const getPackage = (name) => MODELS_ROOT + name;


const _tmpVector3 = new Vector3();
const _tmpQuaternion = new Quaternion();



export default class Robot {
    constructor(name, packagename, xacro) {
        this.name = name;
        this._package = packagename + '/';
        this._xacro = xacro;

        // This will store the model loaded by URDFLoader
        this.model = {}
        this.links = []
        this.joints = []

        // Required for find_package
        this.packages = {
            get [packagename]() {
                return getPackage(packagename);
            }
        };

        // Parts of the arm, filled by init()
        this.arm = {
            joints: [],
            movable: [],
            links: [],
        };

        // Parts of the hand, filled by init()
        this.hand = {
            joints: [],
            movable: [],
            links: [],
            invertOpenClose: false,
        };

        this.robotRoot = "";
        this.handRoot = "";


        /* =============================================================
         * Everything past here should be filled by the deriving classes
         * ============================================================= */

        this.modelScale = 1;

        // Names of links and joints grouped by what they belong to
        this.partNames = {
            arm: [],   // "joint_1", "joint_2", ...
            hand: [],  // "joint_1", "joint_2", ...
        }

        // Default pose of the robot
        this.defaultPose = {
        },

        // Location of the handle used for moving the robot
        this.tcp = {
            parent: "",
            // Distance and euler angles from hand origin to finger tip
            position: [0, 0, 0],
            rotation: [0, 0, 0],
            object: new Object3D(),  // Filled by init()
        };

        // Offsets to apply to the joint position helpers
        this.jointPositionHelperOffsets = {
        };

        // Joint position helpers that must be inverted
        this.jointPositionHelperInverted = [
        ];
    }


    init(model) {
        this.model = model;
        this.joints = model.joints;
        this.links = model.links;

        this.model.scale.set(this.modelScale, this.modelScale, this.modelScale);

        this.tcp.object.position.set(...this.tcp.position);
        this.tcp.object.quaternion.multiply(
            new Quaternion().setFromEuler(
                new Euler().set(...this.tcp.rotation)
            )
        );
        this.getFrame(this.tcp.parent).add(this.tcp.object);

        // Find all joints and links from the robot root up to the hand root
        this.robotRoot = this.getFrame(this.robotRoot);
        traverse(this.robotRoot, (obj) => {
            if (obj.name === this.handRoot) {
                return true;
            }

            this.partNames.arm.push(obj.name);

            if (this.isJoint(obj)) {
                this.arm.joints.push(obj);
                if (this.isMovable(obj)) {
                    this.arm.movable.push(obj);
                }
            }
            else if (this.isLink(obj)) {
                this.arm.links.push(obj);
            }
        });

        // Find all joints and links from the hand root onwards
        this.handRoot = this.getFrame(this.handRoot);
        traverse(this.handRoot, (obj) => {
            this.partNames.hand.push(obj.name);

            if (this.isJoint(obj)) {
                this.hand.joints.push(obj);
                if (this.isMovable(obj)) {
                    this.hand.movable.push(obj);
                }
            }
            else if (this.isLink(obj)) {
                this.hand.links.push(obj);
            }
        });

        for (let joint of this.hand.movable) {
            joint.states = {
                closed: this.hand.invertOpenClose ? joint.limit.upper : joint.limit.lower,
                opened: this.hand.invertOpenClose ? joint.limit.lower : joint.limit.upper,
            };
        }
    }


    get root() {
        return MODELS_ROOT;
    }

    get package() {
        return this.root + this._package;
    }

    get xacro() {
        return this.package + this._xacro;
    }


    setPose(pose) {
        if (Array.isArray(pose)) {
            if (pose.length !== this.arm.movable.length) {
                throw new Error('Array length must be equal to the number of movable joints');
            }

            for (let i = 0; i < pose.length; i++) {
                const joint = this.arm.movable[i];
                joint.setJointValue(pose[i]);
                if (joint.helper != null)
                    joint.helper.updatePosition();
            }
        }
        else if (typeof pose === 'object') {
            for (let joint of this.arm.movable) {
                let value = pose[joint.name];
                if (typeof value === 'number') {
                    joint.setJointValue(value);
                    if (joint.helper != null)
                        joint.helper.updatePosition();
                }
            }
        }
        else {
            throw new Error('Invalid pose type "' + typeof pose + '"');
        }
    }


    getPose() {
        var pose = new Array(this.arm.movable.length);

        for (let i = 0; i < pose.length; i++) {
            pose[i] = this.arm.movable[i].jointValue[0];
        }

        return pose;
    }


    /* Returns the position of the end-effector of the robot (a Vector3)
    */
    getEndEffectorPosition() {
        this.tcp.object.getWorldPosition(_tmpVector3);
        return _tmpVector3.clone();
    }


    /* Returns the orientation of the end-effector of the robot (a Quaternion)
    */
    getEndEffectorOrientation() {
        this.tcp.object.getWorldQuaternion(_tmpQuaternion);
        return _tmpQuaternion.clone();
    }


    /* Returns the position and orientation of the end-effector of the robot in an array
    of the form: [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorTransforms() {
        this.tcp.object.getWorldPosition(_tmpVector3);
        this.tcp.object.getWorldQuaternion(_tmpQuaternion);

        return [
            _tmpVector3.x, _tmpVector3.y, _tmpVector3.z,
            _tmpQuaternion.x, _tmpQuaternion.y, _tmpQuaternion.z, _tmpQuaternion.w,
        ];
    }


    isJoint(part) {
        return part.type === "URDFJoint";
    }

    isLink(part) {
        return part.type === "URDFLink";
    }

    isArm(part) {
        return this.partNames.arm.includes(part.name);
    }

    isHand(part) {
        return this.partNames.hand.includes(part.name);
    }

    isMovable(part) {
        return this.isJoint(part) && part._jointType !== "fixed";
    }


    isGripperOpen() {
        return this.getGripperAbduction() >= 0.5;
    }

    getGripperAbduction() {
        if (this.hand.movable.length === 0) {
            return 0;
        }

        // Average abduction of all hand joints
        let abduction = 0.0;
        for (let joint of this.hand.movable) {
            let val = joint.angle;
            let upper = joint.limit.upper;
            let lower = joint.limit.lower;
            let rel = (val - lower) / (upper - lower);
            abduction += rel;
        }
        abduction /= this.hand.movable.length;
        return abduction;
    }

    closeGripper() {
        const start = {};
        const target = {};

        for (const finger of this.hand.movable) {
            start[finger.name] = finger.angle;
            target[finger.name] = finger.states.closed;
        }

        let tween = new TWEEN.Tween(start)
            .to(target, 1000.0)
            .easing(TWEEN.Easing.Quadratic.Out);

        tween.onUpdate(object => {
            for (const j in object) {
                this.model.joints[j].setJointValue(object[j]);
            }
        });

        tween.start();
    }

    openGripper() {
        const start = {};
        const target = {};

        for (const finger of this.hand.movable) {
            start[finger.name] = finger.angle;
            target[finger.name] = finger.states.opened;
        }

        let tween = new TWEEN.Tween(start)
            .to(target, 1000.0)
            .easing(TWEEN.Easing.Quadratic.Out);

        tween.onUpdate(object => {
            for (const j in object) {
                this.model.joints[j].setJointValue(object[j]);
            }
        });

        tween.start();
    }


    getJointForLink(link) {
        if (typeof link === "string") {
            link = this.model.links[link];
        }

        let p = link.parent;
        while (p) {
            if (this.isJoint(p)) {
                return p;
            }
            p = p.parent;
        }

        return null;
    }

    getLinkForJoint(joint) {
        if (typeof joint === "string") {
            joint = this.model.joints[joint];
        }

        for (let c of joint.children) {
            if (this.isLink(c)) {
                return c;
            }
        }

        return null;
    }

    getFrame(name) {
        return this.model.frames[name];
    }


    createJointPositionHelpers() {
        for (let i = 0; i < this.arm.movable.length; ++i) {
            const joint = this.arm.movable[i];

            const helper = new JointPositionHelper(joint, i + 1, this.jointPositionHelperInverted.includes(joint.name));

            if (this.jointPositionHelperOffsets[joint.name])
                helper.translateZ(this.jointPositionHelperOffsets[joint.name]);
        }
    }


    updatePositionHelpersSize(cameraPosition, elementWidth) {
        for (let i = 0; i < this.arm.movable.length; ++i) {
            const joint = this.arm.movable[i];

            if (joint.helper != null)
                joint.helper.updateSize(cameraPosition, elementWidth);
        }
    }
}
