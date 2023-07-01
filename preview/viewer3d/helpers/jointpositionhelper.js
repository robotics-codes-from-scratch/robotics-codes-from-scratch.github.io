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
import katex from 'katex';


const axisY = new THREE.Vector3(0, 1, 0);


class JointPositionGeometry extends THREE.BufferGeometry {

    constructor(radius = 1, segments = 32, thetaStart = 0, thetaLength = Math.PI * 2) {

        super();

        this.type = 'JointPositionGeometry';

        this.parameters = {
            radius: radius,
            segments: segments,
            thetaStart: thetaStart,
            thetaLength: thetaLength
        };

        this.segments = Math.max(3, segments);

        // buffers
        const indices = [];
        const vertices = [];
        const normals = [];

        // center point
        vertices.push(0, 0, 0);
        normals.push(0, 1, 0);

        for (let s = 0, i = 3; s <= segments; s++, i += 3) {
            // vertex
            vertices.push(0, 0, 0);

            // normal
            normals.push(0, 1, 0);
        }

        // indices
        for (let i = 1; i <= segments; i++)
            indices.push(i, i + 1, 0);

        // position buffer
        this.positionBuffer = new THREE.Float32BufferAttribute(vertices, 3);
        this.positionBuffer.setUsage(THREE.DynamicDrawUsage);

        this.update(thetaStart, thetaLength);

        // build geometry
        this.setIndex(indices);
        this.setAttribute('position', this.positionBuffer);
        this.setAttribute('normal', new THREE.Float32BufferAttribute( normals, 3 ) );
    }

    update(thetaStart = 0, thetaLength = Math.PI * 2) {
        for (let s = 0, i = 3; s <= this.segments; s++, i += 3) {
            const segment = thetaStart + s / this.segments * thetaLength;

            this.positionBuffer.array[i] = this.parameters.radius * Math.cos(segment);
            this.positionBuffer.array[i+2] =  -this.parameters.radius * Math.sin(segment);
        }

        this.positionBuffer.needsUpdate = true;
    }

}



export default class JointPositionHelper extends THREE.Object3D {

    constructor(scene, layer, jointId, jointIndex, jointPosition, invert=false, color=0xff0000, offset=0.0) {
        super();

        this.layers.disableAll();
        this.layers.enable(layer);

        this.isJointPositionHelper = true;
        this.type = 'JointPositionHelper';

        this.jointId = jointId;
        this.invert = invert || false;
        this.previousDistanceToCamera = null;
        this.previousPosition = null;

        this.origin = new THREE.Object3D();
        this.origin.translateY(offset);
        this.add(this.origin);


        if (typeof color === 'string') {
            if (color[0] == '#')
                color = color.substring(1);
            color = Number('0x' + color);
        }

        color = new THREE.Color(color);

        const lineMaterial = new THREE.LineBasicMaterial({
            color: color
        });

        const points = [];
        points.push(new THREE.Vector3(0, 0, 0));
        points.push(new THREE.Vector3(0.2, 0, 0));

        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);


        const circleMaterial = new THREE.MeshBasicMaterial({
            color: color,
            opacity: 0.25,
            transparent: true,
            side: THREE.DoubleSide
        });


        this.startLine = new THREE.Line(lineGeometry, lineMaterial);
        this.startLine.layers = this.layers;
        this.origin.add(this.startLine);

        this.endLine = new THREE.Line(lineGeometry, lineMaterial);
        this.endLine.layers = this.layers;
        this.origin.add(this.endLine);

        if (invert)
            this.endLine.setRotationFromAxisAngle(axisY, Math.PI);

        const circleGeometry = new JointPositionGeometry(0.2, 16, -jointPosition, jointPosition);

        this.circle = new THREE.Mesh(circleGeometry, circleMaterial);
        this.circle.layers = this.layers;
        this.origin.add(this.circle);


        this.labelRotator = new THREE.Object3D();
        this.origin.add(this.labelRotator);

        this.labelElement = document.createElement('div');
        this.labelElement.style.fontSize = '1vw';

        katex.render(String.raw`\color{#` + color.getHexString() + `}x_` + jointIndex, this.labelElement, {
            throwOnError: false
        });

        this.label = new CSS2DObject(this.labelElement);
        this.label.position.set(0.24, 0, 0);
        this.labelRotator.add(this.label);

        this.label.layers.disableAll();
        this.label.layers.enable(31);

        scene.add(this);
    }


    updateTransforms(joint) {
        joint.getWorldPosition(this.position);
        joint.getWorldQuaternion(this.quaternion);
    }


    updateJointPosition(jointPosition) {
        if ((this.previousPosition != null) && (Math.abs(jointPosition - this.previousPosition) < 1e-6))
            return;

        if (this.invert) {
            this.startLine.setRotationFromAxisAngle(axisY, Math.PI - jointPosition);
            this.circle.geometry.update(Math.PI - jointPosition, jointPosition);
            this.labelRotator.setRotationFromAxisAngle(axisY, Math.PI - jointPosition / 2);
        } else {
            this.startLine.setRotationFromAxisAngle(axisY, -jointPosition);
            this.circle.geometry.update(-jointPosition, jointPosition);
            this.labelRotator.setRotationFromAxisAngle(axisY, -jointPosition / 2);
        }

        this.previousPosition = jointPosition;
    }


    updateSize(cameraPosition, elementWidth) {
        const position = new three.Vector3();
        this.getWorldPosition(position);

        const dist = cameraPosition.distanceToSquared(position);

        const maxDist = 0.21 + 0.03 * 1000 / elementWidth;

        if (dist > 30.0) {
            if (this.previousDistanceToCamera != 30) {
                this.labelElement.style.fontSize = '0.7vw';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 30;
            }
        } else if (dist > 10.0) {
            if (this.previousDistanceToCamera != 10) {
                this.labelElement.style.fontSize = '0.8vw';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 10;
            }
        } else if (dist > 5.0) {
            if (this.previousDistanceToCamera != 5) {
                this.labelElement.style.fontSize = '0.9vw';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 5;
            }
        } else {
            if (this.previousDistanceToCamera != 0) {
                this.labelElement.style.fontSize = '1vw';
                this.previousDistanceToCamera = 0;
            }

            this.label.position.x = 0.21 + 0.03 * 1000 / elementWidth * Math.max(dist, 0.001) / 5.0;
        }
    }


    _disableVisibility(materials) {
        this.startLine.material.colorWrite = false;
        this.startLine.material.depthWrite = false;

        this.circle.material.colorWrite = false;
        this.circle.material.depthWrite = false;

        materials.push(this.startLine.material, this.circle.material);
    }

}
