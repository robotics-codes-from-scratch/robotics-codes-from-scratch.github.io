/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


/* Represents a target, an object that can be manipulated by the user that can for example
be used to define a destination position and orientation for the end-effector of the robot.

A target is an Object3D, so you can manipulate it like one.

Note: Targets are top-level objects, so their local position and orientation are also their
position and orientation in world space. 
*/
export default class Target extends THREE.Object3D {

    /* Constructs the 3D viewer

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
    */
    constructor(name, position, orientation, color) {
        super();

        this.name = name;

        // Ensure we have a color
        if (color == null)
            color = 0x0000aa;

        // Create the cone mesh
        const geometry = new THREE.ConeGeometry(0.05, 0.1, 12);

        this.mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({
                color: color,
                opacity: 0.5,
                transparent: true
            })
        );

        this.mesh.quaternion.copy(new THREE.Quaternion(0.707, 0.0, 0.0, 0.707));

        this.mesh.castShadow = true;
        this.mesh.receiveShadow = false;

        this.add(this.mesh);

        // Add a wireframe on top of the cone mesh
        const wireframe = new THREE.WireframeGeometry(geometry);

        let line = new THREE.LineSegments(wireframe);
        line.material.depthTest = true;
        line.material.opacity = 0.5;
        line.material.transparent = true;

        this.mesh.add(line);

        // Set the target position and orientation
        this.position.copy(position);
        this.quaternion.copy(orientation);

        this.mesh.tag = 'target-mesh';
        this.tag = 'target';
    }


    /* Returns the position and orientation of the target in an array of the form:
    [px, py, pz, qx, qy, qz, qw]
    */
    transforms() {
        return [
            this.position.x, this.position.y, this.position.z,
            this.quaternion.x, this.quaternion.y, this.quaternion.z, this.quaternion.w,
        ];
    }
}
