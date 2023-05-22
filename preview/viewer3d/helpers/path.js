/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


/* Visual representation of a path.

A path is an Object3D, so you can manipulate it like one.
*/
export default class Path extends THREE.Object3D {

    /* Constructor

    Parameters:
        name (str): Name of the path
        points (list or Vector3/list of lists of 3 numbers): Points defining the path
        radius (Number): The radius of the path (default is 0.01)
        color (int/str): Color of the path (by default: 0xffff00)
        shading (bool): Indicates if the path must be affected by lights (by default: false)
        transparent (bool): Indicates if the path must be transparent (by default: false)
        opacity (Number): Opacity level for transparent paths (between 0 and 1, default: 0.5)
    */
    constructor(name, points, radius=0.01, color=0xffff00, shading=false, transparent=false, opacity=0.5) {
        super();

        this.name = name;

        let curvePoints = points;
        if (!(points[0] instanceof THREE.Vector3)) {
            curvePoints = points.map(x => new THREE.Vector3(x[0], x[1], x[2]));
        }

        const curve = new THREE.CatmullRomCurve3(curvePoints);

        const geometry = new THREE.TubeGeometry(curve, points.length * 2, radius, 16, false);

        let material;
        if (shading) {
            material = new THREE.MeshPhongMaterial({
                color: color,
                opacity: opacity,
                transparent: transparent
            });
        } else {
            material = new THREE.MeshBasicMaterial({
                color: color,
                opacity: opacity,
                transparent: transparent
            });
        }

        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.layers = this.layers;

        this.add(this.mesh);

        const sphereGeometry = new THREE.SphereGeometry(radius, 16, 16);//, 0, Math.PI);

        this.startSphere = new THREE.Mesh(sphereGeometry, material);
        this.startSphere.position.copy(curvePoints[0]);
        this.startSphere.layers = this.layers;

        this.add(this.startSphere);

        this.endSphere = new THREE.Mesh(sphereGeometry, material);
        this.endSphere.position.copy(curvePoints[curvePoints.length - 1]);
        this.endSphere.layers = this.layers;

        this.add(this.endSphere);
    }


    _disableVisibility(materials) {
        this.mesh.material.colorWrite = false;
        this.mesh.material.depthWrite = false;

        this.startSphere.material.colorWrite = false;
        this.startSphere.material.depthWrite = false;

        this.endSphere.material.colorWrite = false;
        this.endSphere.material.depthWrite = false;

        materials.push(this.mesh.material, this.startSphere.material, this.endSphere.material);
    }

}
