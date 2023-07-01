/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


const Shapes = Object.freeze({
    Cube: Symbol("cube"),
    Cone: Symbol("cone")
});



/* Represents a target, an object that can be manipulated by the user that can for example
be used to define a destination position and orientation for the end-effector of the robot.

A target is an Object3D, so you can manipulate it like one.

Note: Targets are top-level objects, so their local position and orientation are also their
position and orientation in world space. 
*/
class Target extends THREE.Object3D {

    /* Constructs the 3D viewer

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
        shape (Shapes): Shape of the target (by default: Shapes.Cube)
    */
    constructor(name, position, orientation, color=0x0000aa, shape=Shapes.Cube) {
        super();

        this.name = name;

        // Create the mesh
        let geometry = null;
        switch (shape) {
            case Shapes.Cone:
                geometry = new THREE.ConeGeometry(0.05, 0.1, 12);
                break;

            case Shapes.Cube:
            default:
                geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
        }

        this.mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({
                color: color,
                opacity: 0.5,
                transparent: true
            })
        );

        this.mesh.castShadow = true;
        this.mesh.receiveShadow = false;
        this.mesh.layers = this.layers;

        this.add(this.mesh);

        // Add a wireframe on top of the cone mesh
        const wireframe = new THREE.WireframeGeometry(geometry);

        this.line = new THREE.LineSegments(wireframe);
        this.line.material.depthTest = true;
        this.line.material.opacity = 0.5;
        this.line.material.transparent = true;
        this.line.layers = this.layers;

        this.mesh.add(this.line);

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


    _disableVisibility(materials) {
        this.mesh.material.colorWrite = false;
        this.mesh.material.depthWrite = false;

        this.line.material.colorWrite = false;
        this.line.material.depthWrite = false;

        materials.push(this.mesh.material, this.line.material);
    }

}


// Exportations
export { Target, Shapes };
