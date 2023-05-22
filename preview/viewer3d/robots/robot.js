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

import * as THREE from 'three';
import { TWEEN } from 'three/examples/jsm/libs/tween.module.min.js';

import JointPositionHelper from '../helpers/jointpositionhelper.js';


const _tmpVector3 = new THREE.Vector3();
const _tmpQuaternion = new THREE.Quaternion();
const _tmpQuaternion2 = new THREE.Quaternion();


export default class Robot {

    constructor(name, configuration, physicsSimulator) {
        this.name = name;
        this.configuration = configuration;
        this._physicsSimulator = physicsSimulator;

        this.arm = {
            joints: [],
            actuators: [],
            links: [],
            limits: [],

            names: {
                joints: [],
                actuators: [],
                links: [],
            },

            visual: {
                joints: [],
                links: [],
                meshes: [],
                helpers: [],
            },
        };

        this.tool = {
            joints: [],
            actuators: [],
            links: [],

            names: {
                joints: [],
                actuators: [],
                links: [],
            },

            states: [],

            visual: {
                joints: [],
                links: [],
                meshes: [],
            },
        };

        this.tcp = null;
        this.tcpTarget = null;

        this.layers = new THREE.Layers();

        this.savedState = null;
    }


    getJointPositions() {
        return this._physicsSimulator.getJointPositions(this.arm.joints);
    }


    setJointPositions(positions) {
        const pos = positions.map(
            (v, i) => Math.min(Math.max(v, this.arm.limits[i][0]), this.arm.limits[i][1])
        );

        this._physicsSimulator.setJointPositions(pos, this.arm.joints);
        this._physicsSimulator.setControl(pos, this.arm.actuators);
    }


    getControl() {
        return this._physicsSimulator.getControl(this.arm.actuators);
    }


    setControl(control) {
        const ctrl = control.map(
            (v, i) => Math.min(Math.max(v, this.arm.limits[i][0]), this.arm.limits[i][1])
        );

        if (this._physicsSimulator.paused)
            this._physicsSimulator.setJointPositions(ctrl, this.arm.joints);

        this._physicsSimulator.setControl(ctrl, this.arm.actuators);
    }


    getDefaultPose() {
        const pose = new Float32Array(this.arm.joints.length);
        pose.fill(0.0);

        for (let name in this.configuration.defaultPose)
            pose[this.arm.names.joints.indexOf(name)] = this.configuration.defaultPose[name];

        return pose;
    }


    applyDefaultPose() {
        this.setJointPositions(this.getDefaultPose());
    }


    /* Returns the position of the end-effector of the robot (a Vector3)
    */
    getEndEffectorPosition() {
        this.tcp.getWorldPosition(_tmpVector3);
        return _tmpVector3.clone();
    }


    /* Returns the orientation of the end-effector of the robot (a Quaternion)
    */
    getEndEffectorOrientation() {
        this.tcp.getWorldQuaternion(_tmpQuaternion);
        return _tmpQuaternion.clone();
    }


    /* Returns the position and orientation of the end-effector of the robot in an array
    of the form: [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorTransforms() {
        this.tcp.getWorldPosition(_tmpVector3);
        this.tcp.getWorldQuaternion(_tmpQuaternion);

        return [
            _tmpVector3.x, _tmpVector3.y, _tmpVector3.z,
            _tmpQuaternion.x, _tmpQuaternion.y, _tmpQuaternion.z, _tmpQuaternion.w,
        ];
    }


    /* Returns the desired position and orientation for the end-effector of the robot in an
    array of the form: [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulator of the
    end-effector (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorDesiredTransforms() {
        if (this.tcpTarget != null) {
            return [
                this.tcpTarget.position.x, this.tcpTarget.position.y, this.tcpTarget.position.z,
                this.tcpTarget.quaternion.x, this.tcpTarget.quaternion.y, this.tcpTarget.quaternion.z, this.tcpTarget.quaternion.w,
            ];
        }

        return this.robot.getEndEffectorTransforms();
    }


    save() {
        this.savedState = {
            positions: this.getJointPositions(),
            control: this.getControl(),
        };
    }


    restore() {
        if (this.savedState != null) {
            this._physicsSimulator.setJointPositions(this.savedState.positions, this.arm.joints);
            this._physicsSimulator.setControl(this.savedState.control, this.arm.actuators);
            this.savedState = null;
        }
    }


    /* Performs Forward Kinematics, on a subset of the joints

    Parameters:
        positions (array): The joint positions
        offset (Vector3): Optional, an offset from the last joint

    Returns:
        A tuple of a Vector3 and a Quaternion: (position, orientation)
    */
    fkin(positions, offset=null) {
        // Check the input
        if (positions.length > this.arm.joints.length)
            throw new Error('The number of joint positions must be less or equal than the number of movable joints');

        // Determine if we need to save the current state
        const saved = (this.savedState == null);
        if (saved)
            this.save();

        // Set the joint positions
        const limitedPositions = positions.map(
            (v, i) => Math.min(Math.max(v, this.arm.limits[i][0]), this.arm.limits[i][1])
        );

        this._physicsSimulator.setJointPositions(limitedPositions, this.arm.joints.slice(0, limitedPositions.length + 1));
        this._physicsSimulator.simulation.forward();

        // Retrieve the position and orientation of the last joint
        let transforms = null;

        const bodyId = this.arm.visual.joints[positions.length - 1].bodyId;

        const pos = new THREE.Vector3();
        const quat = new THREE.Quaternion();

        if ((positions.length == this.arm.joints.length) && (offset == null)) {
            this._physicsSimulator._getPosition(this._physicsSimulator.simulation.site_xpos, this.tcp.site_id, _tmpVector3);

            const mat3 = new THREE.Matrix3();
            this._physicsSimulator._getMatrix(this._physicsSimulator.simulation.site_xmat, this.tcp.site_id, mat3);

            const mat4 = new THREE.Matrix4();
            mat4.setFromMatrix3(mat3);

            _tmpQuaternion.setFromRotationMatrix(mat4);
        } else {
            this._physicsSimulator._getPosition(this._physicsSimulator.simulation.xpos, bodyId, _tmpVector3);
            this._physicsSimulator._getQuaternion(this._physicsSimulator.simulation.xquat, bodyId, _tmpQuaternion);

            if (offset != null) {
                const offset2 = offset.clone();
                offset2.applyQuaternion(_tmpQuaternion);
                _tmpVector3.add(offset2);
            }
        }

        // Restore the robot state if necessary
        if (saved)
            this.restore();

        return [
            _tmpVector3.x, _tmpVector3.y, _tmpVector3.z,
            _tmpQuaternion.x, _tmpQuaternion.y, _tmpQuaternion.z, _tmpQuaternion.w,
        ];
    }


    isGripperOpen() {
        return this.getGripperAbduction() >= 0.99;
    }


    getGripperAbduction() {
        if (this.tool.actuators.length === 0)
            return 0.0;

        // Average abduction of all tool joints
        let abduction = 0.0;
        const qpos = this._physicsSimulator.getJointPositions(this.tool.joints);

        for (let i = 0; i < this.tool.joints.length; ++i) {
            const joint = this.tool.joints[i];
            const range = this._physicsSimulator.jointRange(joint);
            let rel = (qpos[i] - range[0]) / (range[1] - range[0]);
            abduction += rel;
        }
        abduction /= this.tool.joints.length;

        return abduction;
    }


    closeGripper() {
        this._activateGripper('closed', 0);
    }


    openGripper() {
        this._activateGripper('opened', 1);
    }


    createJointPositionHelpers(scene, layer, colors=[]) {
        const cfg = this.configuration.jointPositionHelpers;
        const x = this.getJointPositions();

        for (let i = 0; i < this.arm.joints.length; ++i) {
            const joint = this.arm.joints[i];
            const name = this.arm.names.joints[i];

            const helper = new JointPositionHelper(
                scene, layer, joint, i + 1, x[i],
                cfg.inverted.includes(name),
                colors[i] || 0xff0000,
                cfg.offsets[name] || 0.0
            );

            helper.updateTransforms(this.arm.visual.joints[i]);

            this.arm.visual.helpers.push(helper);
        }
    }


    synchronize(cameraPosition, elementWidth) {
        // Note: The transforms of the visual representation of the links are already
        // updated by the physics simulator.

        const x = this.getJointPositions();

        for (let i = 0; i < this.arm.visual.helpers.length; ++i) {
            const helper = this.arm.visual.helpers[i];
            helper.updateTransforms(this.arm.visual.joints[i]);
            helper.updateJointPosition(x[i]);
            helper.updateSize(cameraPosition, elementWidth);
        }
    }


    _init() {
        // Retrieve the names of all joints, links and actuators
        this.arm.names.joints = this._physicsSimulator.jointNames(this.arm.joints);
        this.arm.names.actuators = this._physicsSimulator.actuatorNames(this.arm.actuators);
        this.arm.names.links = this._physicsSimulator.bodyNames(this.arm.links);

        this.tool.names.joints = this._physicsSimulator.jointNames(this.tool.joints);
        this.tool.names.actuators = this._physicsSimulator.actuatorNames(this.tool.actuators);
        this.tool.names.links = this._physicsSimulator.bodyNames(this.tool.links);

        // Retrieve the limit of the actuators of the arm
        for (let actuator of this.arm.actuators) {
            const range = this._physicsSimulator.actuatorRange(actuator);
            this.arm.limits.push(range);
        }

        // Retrieve the actuator values representing the states of the tool (if any)
        for (let actuator of this.tool.actuators) {
            const range = this._physicsSimulator.actuatorRange(actuator);

            this.tool.states.push({
                closed: range[0],
                opened: range[1],
            });
        }

        // Retrieve the list of all links (=groups) of by the robot
        this.arm.visual.links = this.arm.links.map((b) => this._physicsSimulator.bodies[b]);

        this.tool.visual.links = this.tool.links.map((b) => this._physicsSimulator.bodies[b]);

        // Retrieve the list of all links (=groups) with a joint of by the robot
        this.arm.visual.joints = this.arm.joints.map((j) => this._physicsSimulator.bodies[this._physicsSimulator.model.jnt_bodyid[j]]);

        this.tool.visual.joints = this.tool.joints.map((j) => this._physicsSimulator.bodies[this._physicsSimulator.model.jnt_bodyid[j]]);

        // Retrieve the list of all the meshes used by the robot
        this.arm.visual.meshes = this.arm.visual.links.map((body) => body.children.filter((c) => c.type == "Mesh")).flat();

        this.tool.visual.meshes = this.tool.visual.links.map((body) => body.children.filter((c) => c.type == "Mesh")).flat();

        for (const mesh of this.arm.visual.meshes)
            mesh.layers = this.layers;

        for (const mesh of this.tool.visual.meshes)
            mesh.layers = this.layers;

        // Apply the default pose defined in the configuration
        this.applyDefaultPose();
    }


    _activateGripper(stateName, rangeIndex) {
        if (this.tool.actuators.length > 0) {
            const ctrl = new Float32Array(this.tool.actuators.length);
            for (let i = 0; i < this.tool.states.length; ++i)
                ctrl[i] = this.tool.states[i][stateName];

            this._physicsSimulator.setControl(ctrl, this.tool.actuators);
        }

        if (this._physicsSimulator.paused) {
            const start = {};
            const target = {};

            const qpos = this._physicsSimulator.getJointPositions(this.tool.joints);

            for (let i = 0; i < this.tool.joints.length; ++i) {
                const name = this.tool.names.joints[i];
                start[name] = qpos[i];
                target[name] = this._physicsSimulator.jointRange(this.tool.joints[i])[rangeIndex];
            }

            let tween = new TWEEN.Tween(start)
                .to(target, 500.0)
                .easing(TWEEN.Easing.Quadratic.Out);

            tween.onUpdate(object => {
                const x = new Float32Array(this.tool.joints.length);

                for (const name in object)
                    x[this.tool.names.joints.indexOf(name)] = object[name];

                this._physicsSimulator.setJointPositions(x, this.tool.joints);
            });

            tween.start();
        }
    }


    _disableVisibility(materials) {
        for (const mesh of this.arm.visual.meshes) {
            const material = mesh.material;
            if (materials.indexOf(material) == -1) {
                material.colorWrite = false;
                material.depthWrite = false;
                materials.push(material);
            }
        }

        for (const mesh of this.tool.visual.meshes) {
            const material = mesh.material;
            if (materials.indexOf(material) == -1) {
                material.colorWrite = false;
                material.depthWrite = false;
                materials.push(material);
            }
        }

        for (const helper of this.arm.visual.helpers)
            helper._disableVisibility(materials);
    }


    _createTcpTarget() {
        this.tcpTarget = new THREE.Mesh(
            new THREE.SphereGeometry(0.2),
            new THREE.MeshBasicMaterial({
                visible: false
            })
        );

        this.tcpTarget.tag = 'tcp-target';

        this.tcp.getWorldPosition(_tmpVector3);
        this.tcp.getWorldQuaternion(_tmpQuaternion);

        this.arm.visual.links[0].worldToLocal(_tmpVector3);
        this.tcpTarget.position.copy(_tmpVector3);

        this.arm.visual.links[0].getWorldQuaternion(_tmpQuaternion2);
        this.tcpTarget.quaternion.multiplyQuaternions(_tmpQuaternion2.invert(), _tmpQuaternion);

        this.arm.visual.links[0].add(this.tcpTarget);
    }

}
