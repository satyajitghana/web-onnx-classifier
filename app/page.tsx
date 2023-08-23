"use client";

import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import { softmax } from "@/utils/math/softmax-2";
import { imagenetClassesTopK } from "@/utils/imagenet";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [resizedImage, setResizedImage] = useState("");
  const [inferenceSession, setInferenceSession] =
    useState<ort.InferenceSession | null>(null);

  const [modelOutput, setModelOutput] = useState<
    { id: string; index: number; name: string; probability: number }[]
  >([]);

  useEffect(() => {
    // load model
    ort.InferenceSession.create("/resnetv2_50.onnx", {
      executionProviders: ["webgl"],
      graphOptimizationLevel: "all",
    }).then((session) => setInferenceSession(session));
  }, []);

  useEffect(() => {
    const mycode = async () => {
      try {
        if (!inferenceSession) return;

        const image = document.createElement("img");
        image.onload = async () => {
          const canvas = document.createElement("canvas");
          canvas.width = 224;
          canvas.height = 224;

          if (!canvas) return;

          const canvas2DCtx = canvas.getContext("2d");

          if (!canvas2DCtx) return;

          canvas2DCtx.drawImage(image, 0, 0, 224, 224);

          const resizedImage = canvas.toDataURL();

          setResizedImage(resizedImage);

          const imageData = canvas2DCtx.getImageData(
            0,
            0,
            canvas2DCtx.canvas.width,
            canvas2DCtx.canvas.height
          );
          const { data, width, height } = imageData;

          // data processing
          const dataTensor = ndarray(new Float32Array(data), [
            width,
            height,
            4,
          ]);

          const dataProcessedTensor = ndarray(
            new Float32Array(width * height * 3),
            [1, 3, width, height]
          );

          // permute [H, W, C] -> [B, C, H, W]
          ops.assign(
            dataProcessedTensor.pick(0, 0, null, null),
            dataTensor.pick(null, null, 0)
          );
          ops.assign(
            dataProcessedTensor.pick(0, 1, null, null),
            dataTensor.pick(null, null, 1)
          );
          ops.assign(
            dataProcessedTensor.pick(0, 2, null, null),
            dataTensor.pick(null, null, 2)
          );

          // image normalization with mean and std
          ops.divseq(dataProcessedTensor, 255);
          ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
          ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
          ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);

          ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
          ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
          ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);

          const tensor = new ort.Tensor(
            "float32",
            new Float32Array(width * height * 3),
            [1, 3, width, height]
          );
          (tensor.data as Float32Array).set(dataProcessedTensor.data);

          // const randomA = Float32Array.from(result);

          // const tensorA = new ort.Tensor("float32", randomA, [1, 3, 224, 224]);

          const results = await inferenceSession.run({
            input: tensor,
          });

          if (results.output) {
            const res = results.output;

            const output = softmax(Array.prototype.slice.call(res.data));
            const topK = imagenetClassesTopK(output, 5);

            setModelOutput(topK);
          }
        };

        if (selectedImage) {
          image.setAttribute("src", URL.createObjectURL(selectedImage));
        }
      } catch (e: any) {
        console.error(e, e.toString());
      }
    };

    if (selectedImage) {
      mycode();
    }
  }, [inferenceSession, selectedImage]);

  return (
    <main className="flex min-h-screen flex-col items-center p-24 gap-y-12">
      {!!!inferenceSession && "Loading Model..."}
      {!!inferenceSession && (
        <input
          type="file"
          name="myImage"
          onChange={(event) => {
            if (event.target.files && event.target.files.length > 0) {
              const file = event.target.files[0];
              console.log(event.target.files[0]);
              setSelectedImage(event.target.files[0]);
            }
          }}
        />
      )}
      {resizedImage && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={resizedImage} alt="Resized Image" className="rounded-md" />
      )}
      {modelOutput.length > 0 && (
        <table className="table-auto max-w-2xl w-full">
          <thead>
            <tr>
              <th className="py-3.5 px-4 text-sm font-normal text-left rtl:text-right text-gray-500 dark:text-gray-400">
                Index
              </th>
              <th className="py-3.5 px-4 text-sm font-normal text-left rtl:text-right text-gray-500 dark:text-gray-400">
                Name
              </th>
              <th className="py-3.5 px-4 text-sm font-normal text-left rtl:text-right text-gray-500 dark:text-gray-400">
                Probability
              </th>
            </tr>
          </thead>
          <tbody>
            {modelOutput.map((m, i) => (
              <tr key={i}>
                <td className="px-4 py-4 text-sm text-gray-500 dark:text-gray-300 whitespace-nowrap">
                  {m.index}
                </td>
                <td className="px-4 py-4 text-sm text-gray-500 dark:text-gray-300 whitespace-nowrap">
                  {m.name}
                </td>
                <td className="px-4 py-4 text-sm text-gray-500 dark:text-gray-300 whitespace-nowrap">
                  {m.probability.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
