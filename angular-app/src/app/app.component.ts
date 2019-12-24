import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';
import * as data  from './labels.json';
import { Tensor, Tensor3D } from '@tensorflow/tfjs';

import { getDataUrlFromArr } from 'array-to-image';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  public webcam = null;
  public model: tf.LayersModel = null;
  public top5Pred = [];
  public labels = (data as any).default;

  public takingPic: boolean = true;
  public screenshotUrl = "";
 
  public screenshot: null;
  private modelName: string = 'dogs-breed-model';

  public ngOnInit(): void {
    this.init();
  }

  async init(){
    try {
      this.webcam = await tfd.webcam(document.getElementById('webcam') as HTMLVideoElement, {'facingMode': 'environment'});
    } catch (e) {
      console.log(e);
      document.getElementById('no-webcam').style.display = 'block';
    }
    await this.loadModel();
  
    const captured = await this.webcam.capture();
    const img = this.preProcessImg(captured);
    this.model.predict(img);
  }

  /**
   * Tries to load keras model from browser indexed db,
   * fallbacks to loading it from backend.
   */
  async loadModel() {
    try {
      console.log("Trying to load model from Indexed DB...");
      this.model = await tf.loadLayersModel('indexeddb://' + this.modelName);
    } catch (e) {
      console.log("Loading from backend...");
      this.model = await tf.loadLayersModel('/assets/models/model.json');
      this.model.save('indexeddb://' + this.modelName);
    }
  }

  /**
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  private preProcessImg(img){
    const processedImg =
        tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    return processedImg;
  }

  /**
   * Captures a picture, pre-processes it and predicts class probabilities.
   * Takes top 5 predictions.
   */
  async predict() {
    const captured = await this.webcam.capture();
    
    this.takingPic = false;

    const img = this.preProcessImg(captured);
    this.imageFromTensor(captured);

    const predictions = this.model.predict(img) as Tensor;
    const indexedPred = this.processPredictions(await predictions.as1D().data());
    this.top5Pred = indexedPred.sort((a, b) =>  b[1] - a[1]).slice(0, 5);
  }

  /**
   * Converts a Tensor3D to a screenshot image.
   */
  private async imageFromTensor(captured: Tensor3D){
    const tensorData = await captured.data();
    const data = new Uint8ClampedArray(299 * 299 * 4);

    for(let i = 0; i < tensorData.length / 3; i += 1) {
      data[4*i] = tensorData[3*i]; // r
      data[4*i + 1] = tensorData[3*i + 1]; // g
      data[4*i + 2] = tensorData[3*i + 2]; // b
      data[4*i + 3] = 255; // a
    }
    this.screenshotUrl = getDataUrlFromArr(data);
  }

  /**
   * Go back to taking pictures.
   */
  public reset() {
    this.takingPic = true;
  }

  /**
   * Preocesses predictions: 
   * maps index to class labels and probabilities to percentages.
   */
  private processPredictions(pred) {
    var result = [];
    pred.forEach((item, index) => {
        result.push([this.labels[index], item*100]);
    });
    return result;
  }

}
