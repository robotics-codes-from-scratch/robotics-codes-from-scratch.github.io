/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


let cylinderGeometry = null;
let coneGeometry = null;
const axis = new THREE.Vector3();


/* Visual representation of an arrow.

An arrow is an Object3D, so you can manipulate it like one.
*/
export default class Arrow extends THREE.Object3D {

    /* Constructor

    Parameters:
        name (str): Name of the arrow
        origin (Vector3): Point at which the arrow starts
        direction (Vector3): Direction from origin (must be a unit vector)
        length (Number): Length of the arrow (default is 1)
        color (int/str): Color of the arrow (by default: 0xffff00)
        shading (bool): Indicates if the arrow must be affected by lights (by default: false)
        headLength (Number): The length of the head of the arrow (default is 0.2 * length)
        headWidth (Number): The width of the head of the arrow (default is 0.2 * headLength)
        radius (Number): The radius of the line part of the arrow (default is 0.1 * headWidth)
    */
    constructor(name, origin, direction, length=1, color=0xffff00, shading=false, headLength=length * 0.2,
        headWidth=headLength * 0.2, radius=headWidth*0.1
    ) {
        super();

        this.name = name;

        if (cylinderGeometry == null) {
            cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 16);
            cylinderGeometry.translate(0, 0.5, 0);

            coneGeometry = new THREE.ConeGeometry(0.5, 1, 16);
            coneGeometry.translate(0, -0.5, 0);
        }

        let material;
        if (shading) {
            material = new THREE.MeshPhongMaterial({
                color: color
            });
        } else {
            material = new THREE.MeshBasicMaterial({
                color: color
            });
        }

        this.cylinder = new THREE.Mesh(cylinderGeometry, material);
        this.cylinder.layers = this.layers;

        this.add(this.cylinder);

        this.cone = new THREE.Mesh(coneGeometry, material);
        this.cone.layers = this.layers;

        this.add(this.cone);

        this.position.copy(origin);
        this.setDirection(direction);
        this.setDimensions(length, headLength, headWidth, radius);
    }


    /* Sets the direction of the arrow
    */
    setDirection(direction) {
        // 'direction' is assumed to be normalized
        if (direction.y > 0.99999) {
            this.quaternion.set(0, 0, 0, 1);
        } else if (direction.y < - 0.99999) {
            this.quaternion.set(1, 0, 0, 0);
        } else {
            axis.set(direction.z, 0, -direction.x).normalize();
            const radians = Math.acos(direction.y);
            this.quaternion.setFromAxisAngle(axis, radians);
        }
    }


    /* Sets the dimensions of the arrow

    Parameters:
        length (Number): Length of the arrow
        color (int/str): Color of the arrow (by default: 0xffff00)
        headLength (Number): The length of the head of the arrow (default is 0.2 * length)
        headWidth (Number): The width of the head of the arrow (default is 0.2 * headLength)
        radius (Number): The radius of the line part of the arrow (default is 0.1 * headWidth)
    */
    setDimensions(length, headLength=length * 0.2, headWidth=headLength * 0.2, radius=headWidth*0.3) {

        this.cylinder.scale.set(
            Math.max(0.0001, radius),
            Math.max(0.0001, length - headLength),
            Math.max(0.0001, radius)
        );

        this.cylinder.updateMatrix();

        this.cone.scale.set(headWidth, headLength, headWidth);
        this.cone.position.y = length;
        this.cone.updateMatrix();
    }


    /* Sets the color of the arrow
    */
    setColor(color) {
        this.line.material.color.set(color);
        this.cone.material.color.set(color);
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        this.cylinder.geometry.dispose();
        this.cone.geometry.dispose();
        this.cone.material.dispose();
    }


    _disableVisibility(materials) {
        this.cylinder.material.colorWrite = false;
        this.cylinder.material.depthWrite = false;

        this.cone.material.colorWrite = false;
        this.cone.material.depthWrite = false;

        materials.push(this.cylinder.material, this.cone.material);
    }
}
