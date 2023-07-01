/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import { Target, Shapes } from './target.js';


export default class TargetList {

    constructor() {
        this.targets = {};
        this.meshes = [];
    }


    /* Create a new target and add it to the list.

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
        shape (Shapes): Shape of the target (by default: Shapes.Cube)

    Returns:
        The target
    */
    create(name, position, orientation, color, shape=Shapes.Cube) {
        const target = new Target(name, position, orientation, color, shape);
        this.add(target);
        return target;
    }


    /* Add a target to the list.

    Parameters:
        target (Target): The target
    */
    add(target) {
        this.targets[target.name] = target;
        this.meshes.push(target.mesh);
    }


    /* Destroy a target.

    Parameters:
        name (str): Name of the target to destroy
    */
    destroy(name) {
        const target = this.targets[name];
        if (target == undefined)
            return;

        if (target.parent != null)
            target.parent.remove(target);

        const index = this.meshes.indexOf(target.mesh);
        this.meshes.splice(index, 1);

        delete this.targets[name];
    }


    /* Returns a target.

    Parameters:
        name (str): Name of the target
    */
    get(name) {
        return this.targets[name] || null;
    }

}
