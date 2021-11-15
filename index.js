import * as tf from '@tensorflow/tfjs';
import * as tfn from "@tensorflow/tfjs-node"
import { get, getSync } from '@andreekeberg/imagedata'
import * as cv from 'canvas'

const handler = tfn.io.fileSystem("C://Object_detection/models-master/research/object_detection/tmp/tfjs_model/model.json");
const model = await tf.loadGraphModel(handler);
const imageFile = "C:/Object_detection/models-master/research/object_detection/test_images/40.jpg"

get(imageFile, (err, imageTensor) => {
    var canvas = cv.createCanvas(imageTensor.width, imageTensor.heigth);
    // myImage.src = imageFile;
    // console.log(myImage)
    // let image = tf.browser.fromPixels(myImage)
    // image = tf.image.resizeBilinear(imageTensor.expandDims().toFloat(), [input_size, input_size]);

    // const predictions = predictLogos(image)
    // console.log(predictions)
})

const ANCHORS = [0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17];

async function predictLogos(inputs) {
	console.log( "Running predictions..." );
	const outputs = await model.executeAsync(inputs, null);
	const arrays = !Array.isArray(outputs) ? outputs.array() : Promise.all(outputs.map(t => t.array()));
	let predictions = await arrays;

	// Post processing for old models.
	if (predictions.length != 3) {
		console.log( "Post processing..." );
	    const num_anchor = ANCHORS.length / 2;
		const channels = predictions[0][0][0].length;
		const height = predictions[0].length;
		const width = predictions[0][0].length;

		const num_class = channels / num_anchor - 5;

		let boxes = [];
		let scores = [];
		let classes = [];

		for (var grid_y = 0; grid_y < height; grid_y++) {
			for (var grid_x = 0; grid_x < width; grid_x++) {
				let offset = 0;

				for (var i = 0; i < num_anchor; i++) {
					let x = (_logistic(predictions[0][grid_y][grid_x][offset++]) + grid_x) / width;
					let y = (_logistic(predictions[0][grid_y][grid_x][offset++]) + grid_y) / height;
					let w = Math.exp(predictions[0][grid_y][grid_x][offset++]) * ANCHORS[i * 2] / width;
					let h = Math.exp(predictions[0][grid_y][grid_x][offset++]) * ANCHORS[i * 2 + 1] / height;

					let objectness = tf.scalar(_logistic(predictions[0][grid_y][grid_x][offset++]));
					let class_probabilities = tf.tensor1d(predictions[0][grid_y][grid_x].slice(offset, offset + num_class)).softmax();
					offset += num_class;

					class_probabilities = class_probabilities.mul(objectness);
					let max_index = class_probabilities.argMax();
					boxes.push([x - w / 2, y - h / 2, x + w / 2, y + h / 2]);
					scores.push(class_probabilities.max().dataSync()[0]);
					classes.push(max_index.dataSync()[0]);
				}
			}
		}

		boxes = tf.tensor2d(boxes);
		scores = tf.tensor1d(scores);
		classes = tf.tensor1d(classes);

		const selected_indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, 10);
		predictions = [await boxes.gather(selected_indices).array(), await scores.gather(selected_indices).array(), await classes.gather(selected_indices).array()];
	}

	return predictions;
}

function _logistic(x) {
	if (x > 0) {
	    return (1 / (1 + Math.exp(-x)));
	} else {
	    const e = Math.exp(x);
	    return e / (1 + e);
	}
}