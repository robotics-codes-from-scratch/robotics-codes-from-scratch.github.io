/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';


const axisZ = new THREE.Vector3(0, 0, 1);


export default class JointPositionHelper extends THREE.Object3D {

    constructor(joint, jointIndex, invert) {
        super();

        this.isJointPositionHelper = true;
        this.type = 'JointPositionHelper';

        this.joint = joint;
        this.invert = invert || false;
        this.previousDistanceToCamera = null;


        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0xff0000
        });

        const points = [];
        points.push(new THREE.Vector3(0, 0, 0));
        points.push(new THREE.Vector3(0.2, 0, 0));

        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);


        const circleMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            opacity: 0.25,
            transparent: true,
            side: THREE.DoubleSide
        });


        this.startLine = new THREE.Line(lineGeometry, lineMaterial);
        this.add(this.startLine);

        this.endLine = new THREE.Line(lineGeometry, lineMaterial);
        this.add(this.endLine);

        if (invert)
            this.endLine.setRotationFromAxisAngle(axisZ, Math.PI);

        const circleGeometry = new THREE.CircleGeometry(0.2, 16, -joint.jointValue[0], joint.jointValue[0]);

        this.circle = new THREE.Mesh(circleGeometry, circleMaterial);
        this.add(this.circle);


        this.labelRotator = new THREE.Object3D();
        this.add(this.labelRotator);

        this.labelElement = document.createElement('div');
        this.labelElement.className = 'joint-label';

        let span = document.createElement('span');
        span.textContent = 'x';
        this.labelElement.appendChild(span);

        span = document.createElement('span');
        span.className = 'sub';
        span.textContent = '' + jointIndex;
        this.labelElement.appendChild(span);

        this.label = new CSS2DObject(this.labelElement);
        this.label.position.set(0.24, 0, 0);
        this.labelRotator.add(this.label);

        joint.add(this);
        joint.helper = this;

        this.updatePosition();
    }


    updatePosition() {
        let position = this.joint.jointValue[0];

        if (this.invert) {
            this.startLine.setRotationFromAxisAngle(axisZ, Math.PI - position);
            this.circle.geometry = new three.CircleGeometry(0.2, 16, Math.PI - position, position);
            this.labelRotator.setRotationFromAxisAngle(axisZ, Math.PI - position / 2);
        } else {
            this.startLine.setRotationFromAxisAngle(axisZ, -position);
            this.circle.geometry = new three.CircleGeometry(0.2, 16, -position, position);
            this.labelRotator.setRotationFromAxisAngle(axisZ, -position / 2);
        }
    }


    updateSize(cameraPosition, elementWidth) {
        const position = new three.Vector3();
        this.getWorldPosition(position);

        const dist = cameraPosition.distanceToSquared(position);

        const maxDist = 0.21 + 0.03 * 1000 / elementWidth;

        if (dist > 30.0) {
            if (this.previousDistanceToCamera != 30) {
                this.labelElement.style.fontSize = 'x-small';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 30;
            }
        } else if (dist > 10.0) {
            if (this.previousDistanceToCamera != 10) {
                this.labelElement.style.fontSize = 'small';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 10;
            }
        } else if (dist > 5.0) {
            if (this.previousDistanceToCamera != 5) {
                this.labelElement.style.fontSize = 'medium';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 5;
            }
        } else {
            if (this.previousDistanceToCamera != 0) {
                this.labelElement.style.fontSize = 'large';
                this.previousDistanceToCamera = 0;
            }

            this.label.position.x = 0.21 + 0.03 * 1000 / elementWidth * Math.max(dist, 0.001) / 5.0;
        }
    }

}
