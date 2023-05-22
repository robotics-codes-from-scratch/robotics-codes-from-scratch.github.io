/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


export default class PartialIKControls {

    constructor(ikPartialFunction) {
        this.robot = null;
        this.offset = null;
        this.joint = null;

        this.plane = new THREE.Mesh(
            new THREE.PlaneGeometry(100000, 100000, 2, 2),
            new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide})
        );

        this.ikPartialFunction = ikPartialFunction
    }


    setup(robot, offset, jointIndex, startPosition, planeDirection) {
        this.robot = robot;
        this.offset = offset;
        this.joint = jointIndex;

        this.plane.position.copy(startPosition); 

        this.plane.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), planeDirection);
        this.plane.updateMatrixWorld(true);
    }


    process(raycaster) {
        let intersects = raycaster.intersectObject(this.plane, false);

        if (intersects.length > 0)
            this.ikPartialFunction(this.robot, intersects[0].point, this.joint, this.offset);
    }

}
