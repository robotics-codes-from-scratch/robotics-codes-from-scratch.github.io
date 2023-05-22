/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


/* A simplified representation of a robot, containing only joints, to be used to compute
  Forward Kinematics without affecting the rendering
*/
export default class FKRobot extends THREE.Object3D {

    /* Construct a simplified representation of the provided robot

    Parameters:
        robot (Robot): The robot to use as a reference
    */
    constructor(robot) {
        super();

        const urdfRobot = robot.arm.links.filter(link => link.isURDFRobot)[0];

        urdfRobot.getWorldPosition(this.position);
        urdfRobot.getWorldQuaternion(this.quaternion);
        
        this.joints = new Array(robot.arm.movable.length);
        this.tcp = null;
        
        this._copyChildJoints(robot, urdfRobot, this);
    }


    /* Performs Forward Kinematics

    Parameters:
        positions (array): The joint positions

    Returns:
        A tuple of a Vector3 and a Quaternion: (position, orientation)
    */
    fkin(positions) {
        if (Array.isArray(positions)) {
            if (positions.length !== this.joints.length) {
                throw new Error('Array length must be equal to the number of movable joints');
            }

            for (let i = 0; i < positions.length; i++) {
                this.joints[i].setJointValue(positions[i]);
            }

            const result = [
                new THREE.Vector3(),
                new THREE.Quaternion(),
            ];

            this.tcp.getWorldPosition(result[0]);
            this.tcp.getWorldQuaternion(result[1]);

            return result;
        }
        else {
            throw new Error('Invalid positions type "' + typeof positions + '"');
        }
    }


    /* Performs Forward Kinematics, on a subset of the joints

    Parameters:
        positions (array): The joint positions
        offset (Vector3): Optional, an offset from the last joint

    Returns:
        A tuple of a Vector3 and a Quaternion: (position, orientation)
    */
    fkinPartial(positions, offset=null) {
        if (Array.isArray(positions)) {
            if (positions.length > this.joints.length) {
                throw new Error('Array length must be less or equal than the number of movable joints');
            }

            for (let i = 0; i < positions.length; i++) {
                this.joints[i].setJointValue(positions[i]);
            }

            const result = [
                new THREE.Vector3(),
                new THREE.Quaternion(),
            ];

            const lastJoint = this.joints[positions.length - 1];

            if (offset != null) {
                if (Array.isArray(offset))
                    offset = new three.Vector3(...offset);

                lastJoint.updateWorldMatrix(true, false);
                result[0].copy(lastJoint.localToWorld(offset));
            } else {
                lastJoint.getWorldPosition(result[0]);
            }

            lastJoint.getWorldQuaternion(result[1]);

            return result;
        }
        else {
            throw new Error('Invalid positions type "' + typeof positions + '"');
        }
    }


    _copyChildJoints(robot, srcParent, dstParent) {
        srcParent.children.forEach(child => {
            if (child.isURDFJoint) {
                let joint = new child.constructor();
                joint.copy(child, false);

                dstParent.add(joint);

                const index = robot.arm.movable.indexOf(child);
                if (index != -1)
                    this.joints[index] = joint;

                this._copyChildJoints(robot, child, joint);
            } else if (Object.is(child, robot.tcp.object)) {
                this.tcp = new THREE.Object3D();
                this.tcp.copy(child, false);
                
                dstParent.add(this.tcp);
            } else {
                this._copyChildJoints(robot, child, dstParent);
            }
        });
    }
}
