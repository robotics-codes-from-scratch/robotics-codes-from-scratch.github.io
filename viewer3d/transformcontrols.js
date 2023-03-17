/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { getURL } from './utils.js';


/* Provides controls to translate/rotate an object, either in world or local coordinates.

Buttons are displayed when a transformation is in progress, to switch between
translation/rotation and world/local coordinates.
*/
export default class TransformControlsManager {

    /* Construct the manager.

    Parameters:
        domElement (element): The DOM element used by the 3D viewer
        rendererElement (element): The DOM element used by the renderer
        camera (Camera): The camera used to render the scene
        scene (Scene): The scene containing the objects to manipulate
    */
    constructor(domElement, rendererElement, camera, scene) {
        this.transformControls = new TransformControls(camera, rendererElement);
        scene.add(this.transformControls);

        this.buttonsContainer = null;
        this.btnTranslation = null;
        this.btnRotation = null;
        this.btnWorld = null;
        this.btnLocal = null;
        
        this.enabled = true;

        this._createButtons(domElement);
        this.enable(false);
    }


    /* Sets up a function that will be called whenever the specified event happens
    */
    addEventListener(name, fct) {
        this.transformControls.addEventListener(name, fct);
    }


    /* Enables/disables the controls
    */
    enable(enabled) {
        this.enabled = enabled && (this.transformControls.object != null);

        if (this.enabled) {
            this.buttonsContainer.style.display = 'block';
            this.transformControls.visible = true;
        } else {
            this.buttonsContainer.style.display = 'none';
            this.transformControls.visible = false;
        }
    }


    /* Indicates whether or not the controls are enabled
    */
    isEnabled() {
        return this.enabled;
    }


    /* Indicates whether or not dragging is currently performed
    */
    isDragging() {
        return this.transformControls.dragging;
    }


    /* Sets the 3D object that should be transformed and ensures the controls UI is visible.

    Parameters:
        object (Object3D): The 3D object that should be transformed
    */
    attach(object) {
        this.transformControls.attach(object);
        this.enable(true);
    }


    /* Removes the current 3D object from the controls and makes the helper UI invisible.
    */
    detach() {
        this.transformControls.detach();
        this.enable(false);
    }


    _createButtons(domElement) {
        this._addCSS();

        this.buttonsContainer = document.createElement('div');
        this.buttonsContainer.className = 'buttons-container';
        domElement.insertBefore(this.buttonsContainer, domElement.firstChild);

        this.btnTranslation = document.createElement('button');
        this.btnTranslation.innerText = 'Translation';
        this.btnTranslation.className = 'left activated';
        this.buttonsContainer.appendChild(this.btnTranslation);

        this.btnRotation = document.createElement('button');
        this.btnRotation.innerText = 'Rotation';
        this.btnRotation.className = 'right';
        this.buttonsContainer.appendChild(this.btnRotation);
    
        this.btnWorld = document.createElement('button');
        this.btnWorld.innerText = 'World';
        this.btnWorld.className = 'left spaced activated';
        this.buttonsContainer.appendChild(this.btnWorld);

        this.btnLocal = document.createElement('button');
        this.btnLocal.innerText = 'Local';
        this.btnLocal.className = 'right';
        this.buttonsContainer.appendChild(this.btnLocal);

        this.btnTranslation.addEventListener('click', evt => this._onTranslationButtonClicked(evt));
        this.btnRotation.addEventListener('click', evt => this._onRotationButtonClicked(evt));
        this.btnWorld.addEventListener('click', evt => this._onWorldButtonClicked(evt));
        this.btnLocal.addEventListener('click', evt => this._onLocalButtonClicked(evt));
    }


    _addCSS() {
        var link = document.createElement('link');
        link.rel = 'stylesheet';
        link.type = 'text/css';
        link.href = getURL('css/style.css');
        document.getElementsByTagName('HEAD')[0].appendChild(link);
    }


    _onTranslationButtonClicked(event) {
        if (this.transformControls.mode == 'translate')
            return;

        this.btnRotation.classList.remove('activated');
        this.btnTranslation.classList.add('activated');
        this.transformControls.setMode('translate');
    }


    _onRotationButtonClicked(event) {
        if (this.transformControls.mode == 'rotate')
            return;

        this.btnTranslation.classList.remove('activated');
        this.btnRotation.classList.add('activated');
        this.transformControls.setMode('rotate');
    }


    _onWorldButtonClicked(event) {
        if (this.transformControls.space == 'world')
            return;

        this.btnLocal.classList.remove('activated');
        this.btnWorld.classList.add('activated');
        this.transformControls.setSpace('world');
    }


    _onLocalButtonClicked(event) {
        if (this.transformControls.space == 'local')
            return;

        this.btnWorld.classList.remove('activated');
        this.btnLocal.classList.add('activated');
        this.transformControls.setSpace('local');
    }
}
