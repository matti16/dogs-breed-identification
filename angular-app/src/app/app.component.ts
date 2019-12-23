import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';
import * as data  from './labels.json';
import { Tensor } from '@tensorflow/tfjs';

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
  
    await this.predict();
  }

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
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  async getImage() {
    const img = await this.webcam.capture();
    const processedImg =
        tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
  }

  async predict() {

    while (true) {
      // Capture the frame from the webcam.
      const img = await this.getImage();
  
      const predictions = this.model.predict(img) as Tensor;
      const indexedPred = this.zipWithIndex(await predictions.as1D().data());
      this.top5Pred = indexedPred.sort((a, b) =>  b[1] - a[1]).slice(0, 5);
      
      img.dispose();
      
      await tf.nextFrame();
    }
  }

  zipWithIndex(list) {
    var result = [];
    list.forEach((item, index) => {
        result.push([this.labels[index], item*100]);
    });
    return result;
  }

}
