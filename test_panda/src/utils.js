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


export { traverse }