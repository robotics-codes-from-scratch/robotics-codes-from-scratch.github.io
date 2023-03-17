/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


export default class ArrowList {

    constructor() {
        this.arrows = {};
    }


    /* Create a new arrow and add it to the list.

    Parameters:
        name (str): Name of the arrow
        origin (Vector3): Point at which the arrow starts
        direction (Vector3): Direction from origin (must be a unit vector)
        length (Number): Length of the arrow (default is 1)
        color (int/str): Color of the arrow (by default: 0xffff00)
        headLength (Number): The length of the head of the arrow (default is 0.2 * length)
        headWidth (Number): The width of the head of the arrow (default is 0.2 * headLength)

    Returns:
        The arrow (THREE.ArrowHelper)
    */
    create(name, origin, direction, length, color, headLength, headWidth) {
        const arrow = new THREE.ArrowHelper(direction, origin, length, color, headLength, headWidth);
        arrow.name = name;

        this.arrows[name] = arrow;

        return arrow;
    }


    /* Destroy an arrow.

    Parameters:
        name (str): Name of the arrow to destroy
    */
    destroy(name) {
        const arrow = this.arrows[name];
        if (arrow == undefined)
            return;

        if (arrow.parent != null)
            arrow.parent.remove(arrow);

        arrow.dispose();

        delete this.arrows[name];
    }


    /* Returns an arrow.

    Parameters:
        name (str): Name of the arrow
    */
    get(name) {
        return this.arrows[name] || null;
    }

}
