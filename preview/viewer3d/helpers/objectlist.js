/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


export default class ObjectList {

    constructor() {
        this.objects = {};
    }


    /* Add an object to the list.

    Parameters:
        object (Object3D): The object
    */
    add(object) {
        this.objects[object.name] = object;
    }


    /* Destroy an object.

    Parameters:
        name (str): Name of the object to destroy
    */
    destroy(name) {
        const object = this.objects[name];
        if (object == undefined)
            return;

        if (object.parent != null)
            object.parent.remove(object);

        delete this.objects[name];
    }


    /* Returns an object.

    Parameters:
        name (str): Name of the object
    */
    get(name) {
        return this.objects[name] || null;
    }

}
