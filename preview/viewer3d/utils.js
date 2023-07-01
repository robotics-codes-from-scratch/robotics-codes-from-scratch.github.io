/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

// A version of traverse that will stop a branch when the callback returns "true"
function traverse(obj, cb, descendants = "children") {
    let ret = cb(obj);

    if (ret || !obj[descendants]) {
        return;
    }

    const children = obj[descendants];
    for (let i = 0; i < children.length; i++) {
        traverse(children[i], cb, descendants);
    }
}


function getURL(path) {
    let url = new URL(import.meta.url);
    return url.href.substring(0, url.href.lastIndexOf('/')) + '/' + path;
}


export { traverse, getURL }
