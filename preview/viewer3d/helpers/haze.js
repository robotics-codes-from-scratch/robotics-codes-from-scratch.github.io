/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


/* Truncated cone for haze rendering
*/
class HazeGeometry extends THREE.BufferGeometry {

    constructor(nbSlices, proportion, color) {
        super();

        this.type = 'HazeGeometry';

        // buffers
        const indices = [];
        const vertices = [];
        const colors = [];

        // Compute elevation h for transparancy transition point
        const alpha = Math.atan2(1.0, proportion);
        const beta = (0.75 * Math.PI) - alpha;
        const h = Math.sqrt(0.5) * proportion * Math.sin(alpha) / Math.sin(beta);

        for (let i = 0; i < 2; ++i) {
            const h1 = (i == 0 ? 0 : h);
            const h2 = (i == 0 ? h : 1);

            const alpha1 = (i == 1);
            const alpha2 = (i == 0);

            for (let j = 0; j < nbSlices; ++j) {
                const az1 = (2.0 * Math.PI * (j + 0)) / nbSlices;
                const az2 = (2.0 * Math.PI * (j + 1)) / nbSlices;

                const index = vertices.length / 3;

                this._makeVertex(vertices, az1, h1, proportion);
                this._makeVertex(vertices, az2, h1, proportion);
                this._makeVertex(vertices, az2, h2, proportion);
                this._makeVertex(vertices, az1, h2, proportion);

                colors.push(color.r, color.g, color.b, alpha1);
                colors.push(color.r, color.g, color.b, alpha1);
                colors.push(color.r, color.g, color.b, alpha2);
                colors.push(color.r, color.g, color.b, alpha2);

                indices.push(index, index + 1, index + 2);
                indices.push(index + 2, index + 3, index);
            }
        }

        // build geometry
        this.setIndex(indices);
        this.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        this.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));
    }

    _makeVertex(vertices, az, h, r) {
        vertices.push(
            Math.cos(az) * (1.0 - r * (1.0 - h)),
            h,
            -Math.sin(az) * (1.0 - r * (1.0 - h))
        );
    }

}



/* Truncated cone for haze rendering
*/
export default class Haze extends THREE.Object3D {

    constructor(nbSlices, proportion, color) {
        super();

        const geometry = new HazeGeometry(nbSlices, proportion, color);

        const material = new THREE.MeshBasicMaterial({
            transparent: true,
            vertexColors: true,
            side: THREE.BackSide
        });

        this.mesh = new THREE.Mesh(geometry, material);
        this.add(this.mesh);
    }

}
