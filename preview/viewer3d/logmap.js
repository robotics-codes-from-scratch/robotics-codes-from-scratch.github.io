/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';


/* Class in charge of the rendering of a logmap distance on a sphere.

It works by first rendering the sphere, the plane and the various points and lines in a
render target. Then the render target's texture is used in a sprite located in the upper
left corner, in a scene using an orthographic camera.

It is expected that the caller doesn't clear the color buffer (but only its depth buffer),
but render its content on top of it (without any background), after calling the 'render()'
method of the logmap object.
*/
export default class Logmap {

    /* Constructs the logmap visualiser

    Parameters:
        domElement (element): The DOM element used by the 3D viewer
        size (int): Size of the sphere (in pixels, approximately. Default: 1/10 of the
                    width of the DOM element at creation time)
    */
    constructor(domElement, robot, target, size=null, position='left') {
        this.domElement = domElement;
        this.robot = robot;
        this.target = target;

        this.scene = null;
        this.orthoScene = null;

        this.camera = null;
        this.orthoCamera = null;

        this.render_target = null;

        this.size = (size || Math.round(this.domElement.clientWidth * 0.1)) * 5;
        this.textureSize = this.size * window.devicePixelRatio;

        this.position = position;

        this.sphere = null;
        this.destPoint = null;
        this.destPointCtrl = null;
        this.srcPoint = null;
        this.srcPointCtrl = null;
        this.projectedPoint = null;
        this.line = null;
        this.plane = null;
        this.sprite = null;

        this._initScene();
    }


    /* Render the background and the logmap

    Parameters:
        renderer (WebGLRenderer): The renderer to use
        cameraOrientation (Quaternion): The orientation of the camera (the logmap sphere will
                                        be rendered using a camera with this orientation, in
                                        order to rotate along the user camera)
    */
    render(renderer, cameraOrientation) {
        // Update the size of the render target if necessary
        if (this.textureSize != this.size * window.devicePixelRatio) {
            this.textureSize = this.size * window.devicePixelRatio;
            this.render_target.setSize(this.textureSize, this.textureSize);
        }

        // Update the logmap using the orientations of the TCP and the target
        this._update(this.target.quaternion, this.robot.getEndEffectorOrientation());

        // Render into the render target
        this.camera.position.x = 0;
        this.camera.position.y = 0;
        this.camera.position.z = 0;
        this.camera.setRotationFromQuaternion(cameraOrientation);
        this.camera.translateZ(10);

        renderer.setRenderTarget(this.render_target);
        renderer.setClearColor(new THREE.Color(0.0, 0.0, 0.0), 0.0);
        renderer.clear();
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(null);

        // Render into the DOM element
        renderer.render(this.orthoScene, this.orthoCamera);
    }


    _initScene() {
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;

        // Cameras
        this.camera = new THREE.PerspectiveCamera(45, 1.0, 0.1, 2000);

        this.orthoCamera = new THREE.OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, -10, 10);
        this.orthoCamera.position.z = 10;

        // Render target
        this.render_target = new THREE.WebGLRenderTarget(
            this.textureSize, this.textureSize,
            {
                encoding: THREE.sRGBEncoding
            }
        );

        // Scenes
        this.scene = new THREE.Scene();
        this.orthoScene = new THREE.Scene();

        // Sphere
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 16);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x156289,
            emissive: 0x072534,
            side: THREE.FrontSide,
            flatShading: false,
            opacity: 0.75,
            transparent: true
        });
        this.sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.scene.add(this.sphere);

        // Points
        const pointGeometry = new THREE.CircleGeometry(0.05, 12);

        const destPointMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            side: THREE.DoubleSide,
        });

        const srcPointMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            side: THREE.DoubleSide,
        });

        const projectedPointMaterial = new THREE.MeshBasicMaterial({
            color: 0xffff00,
            side: THREE.DoubleSide,
        });

        this.destPoint = new THREE.Mesh(pointGeometry, destPointMaterial);
        this.destPoint.position.y = 1.0;
        this.destPoint.rotateX(-Math.PI / 2.0);

        this.destPointCtrl = new THREE.Object3D();
        this.destPointCtrl.add(this.destPoint);
        this.scene.add(this.destPointCtrl);

        this.srcPoint = new THREE.Mesh(pointGeometry, srcPointMaterial);
        this.srcPoint.position.y = 1.0;
        this.srcPoint.rotateX(-Math.PI / 2.0);

        this.srcPointCtrl = new THREE.Object3D();
        this.srcPointCtrl.add(this.srcPoint);
        this.scene.add(this.srcPointCtrl);

        this.projectedPoint = new THREE.Mesh(pointGeometry, projectedPointMaterial);
        this.scene.add(this.projectedPoint);

        // Projected line
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0xe97451,
        });

        const points = [new THREE.Vector3(), new THREE.Vector3()];
        this.destPoint.getWorldPosition(points[0]);
        this.projectedPoint.getWorldPosition(points[1]);

        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);

        this.line = new THREE.Line(lineGeometry, lineMaterial);
        this.scene.add(this.line);

        // Plane
        const planeGeometry = new THREE.PlaneGeometry(3, 3);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide,
            opacity: 0.3,
            transparent: true
        });

        this.plane = new THREE.Mesh(planeGeometry, planeMaterial);
        this.destPoint.add(this.plane);

        // Lights
        const light = new THREE.HemisphereLight(0xffeeee, 0x111122);
        this.scene.add(light);

        const pointLight = new THREE.PointLight(0xffffff, 0.3);
        pointLight.position.set(3, 3, 4);
        this.scene.add(pointLight);

        // Sprite in the final scene
        const spriteMaterial = new THREE.SpriteMaterial({
            map: this.render_target.texture,
        });
        this.sprite = new THREE.Sprite(spriteMaterial);
        this.orthoScene.add(this.sprite);

        this._updateSpritePosition();

        // Events handling
        new ResizeObserver(() => this._onDomElementResized()).observe(this.domElement)
    }


    _onDomElementResized() {
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;

        this.orthoCamera.left = -width / 2;
        this.orthoCamera.right = width / 2;
        this.orthoCamera.top = height / 2;
        this.orthoCamera.bottom = -height / 2;
        this.orthoCamera.updateProjectionMatrix();

        this._updateSpritePosition();
    }


    _updateSpritePosition() {
        const halfWidth = this.domElement.clientWidth / 2;
        const halfHeight = this.domElement.clientHeight / 2;
        const margin = 10;

        if (this.position == 'right') {
            this.sprite.position.set(halfWidth - this.size / 8 - margin, halfHeight - this.size / 8 - margin, 1);
        } else {
            this.sprite.position.set(-halfWidth + this.size / 8 + margin, halfHeight - this.size / 8 - margin, 1);
        }

        this.sprite.scale.set(this.size, this.size, 1);
    }


    _update(mu, f) {
        this.destPointCtrl.setRotationFromQuaternion(mu);
        this.srcPointCtrl.setRotationFromQuaternion(f);

        const base = new THREE.Vector3();
        const y = new THREE.Vector3();

        this.destPoint.getWorldPosition(base);
        this.srcPoint.getWorldPosition(y);

        const temp = y.clone().sub(base.clone().multiplyScalar(base.dot(y)));
        if (temp.lengthSq() > 1e-9) {
            temp.normalize();
            this.projectedPoint.position.addVectors(base, temp.multiplyScalar(this._distance(base, y)));
        } else {
            this.projectedPoint.position.copy(base);
        }

        this.projectedPoint.position.addVectors(base, temp);
        this.destPoint.getWorldQuaternion(this.projectedPoint.quaternion);

        const points = [new THREE.Vector3(), new THREE.Vector3()];
        this.destPoint.getWorldPosition(points[0]);
        this.projectedPoint.getWorldPosition(points[1]);

        this.line.geometry.setFromPoints(points);
    }


    _distance(x, y) {
        let dist = x.dot(y);

        if (dist > 1.0) {
            dist = 1.0;
        } else if (dist < -1.0) {
            dist = -1.0;
        }

        return Math.acos(dist);
    }
}
