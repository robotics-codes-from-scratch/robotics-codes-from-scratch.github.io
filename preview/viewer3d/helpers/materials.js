/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import { DataTexture, MeshToonMaterial, RedFormat } from "three";


const colors = new Uint8Array(3);

for (let c = 0; c <= colors.length; c++)
    colors[c] = (c / colors.length) * 128;

const gradientMap = new DataTexture(colors, colors.length, 1, RedFormat);
gradientMap.needsUpdate = true;


function enableToonShading(object) {
    if (object.isMesh) {
        object.material = new MeshToonMaterial({
            color: object.material.color,
            gradientMap: gradientMap,
        });
    } else {
        object.children.forEach(child => { enableToonShading(child) });
    }
}


function enableLightToonShading(object) {
    if (object.isMesh) {
        object.material.color.r = object.material.color.r * 0.6;
        object.material.color.g = object.material.color.g * 0.6;
        object.material.color.b = object.material.color.b * 0.6;

        object.material = new MeshToonMaterial({
            color: object.material.color,
            emissive: 0x777777,
            gradientMap: gradientMap,
        });
    } else {
        object.children.forEach(child => { enableLightToonShading(child) });
    }
}


// Exportations
export { enableToonShading, enableLightToonShading };
