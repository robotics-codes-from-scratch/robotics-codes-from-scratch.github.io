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


/* Visual representation of a path.

A path is an Object3D, so you can manipulate it like one.
*/
export default class Point extends THREE.Object3D {

    /* Constructor

    Parameters:
        name (str): Name of the point
        position (Vector3): Position of the point
        radius (Number): The radius of the point (default is 0.01)
        color (int/str): Color of the point (by default: 0xffff00)
        label (str): LaTeX text to display near the point (by default: null)
        shading (bool): Indicates if the point must be affected by lights (by default: false)
        transparent (bool): Indicates if the point must be transparent (by default: false)
        opacity (Number): Opacity level for transparent points (between 0 and 1, default: 0.5)
    */
    constructor(name, position, radius=0.01, color=0xffff00, label=null, shading=false,
                transparent=false, opacity=0.5) {
        super();

        this.name = name;
        this.position.copy(position);

        if (typeof(color) == 'string')
            if (color[0] == '#')
                color = color.substring(1);
            color = Number('0x' + color);

        color = new THREE.Color(color);

        const geometry = new THREE.SphereGeometry(radius);

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

        if (label != null) {
            this.labelElement = document.createElement('div');
            this.labelElement.style.fontSize = '1vw';

            katex.render(String.raw`\color{#` + color.getHexString() + `}` + label, this.labelElement, {
                throwOnError: false
            });

            this.label = new CSS2DObject(this.labelElement);
            this.label.position.set(0,2 * radius + 0.01, 0);

            this.label.layers.disableAll();
            this.label.layers.enable(31);

            this.add(this.label);
        }
    }


    _disableVisibility(materials) {
        this.mesh.material.colorWrite = false;
        this.mesh.material.depthWrite = false;

        materials.push(this.mesh.material);
    }

}
