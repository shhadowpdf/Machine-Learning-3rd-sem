import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [allProbs, setAllProbs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleUpload = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setImagePreview(URL.createObjectURL(selected));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select an image first.");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);

      if (data.all_probs) {
        const chartData = Object.entries(data.all_probs)
          .map(([label, prob]) => ({
            label,
            value: (prob * 100).toFixed(2),
          }))
          .sort((a, b) => b.value - a.value);
        setAllProbs(chartData);
      }

      setHistory((prev) => [
        {
          name: data.disease,
          confidence: (data.confidence * 100).toFixed(2),
          time: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 4), // keep last 5
      ]);
    } catch (err) {
      console.error(err);
      alert("Error while analyzing. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ["#22c55e", "#16a34a", "#4ade80", "#86efac", "#15803d"];

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-green-100 py-10 px-4 flex flex-col items-center">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-green-700 mb-2">
          🌿 Crop Disease Detector
        </h1>
        <p className="text-gray-600 text-sm md:text-base">
          Upload a leaf image to identify plant diseases and receive actionable
          care advice.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded-2xl shadow-md w-full max-w-xl"
      >
        <label className="block mb-4 text-left">
          <span className="text-gray-700 font-medium">Select Leaf Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={handleUpload}
            className="mt-2 block w-full text-sm text-gray-600 
                      file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0
                      file:text-sm file:font-semibold file:bg-green-100 
                      file:text-green-700 hover:file:bg-green-200"
          />
        </label>

        {imagePreview && (
          <div className="flex justify-center mb-4">
            <img
              src={imagePreview}
              alt="Preview"
              className="rounded-xl w-64 h-64 object-cover border border-gray-200 shadow-sm"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className={`w-full py-2 px-4 rounded-md text-white font-semibold
                      ${loading ? "bg-gray-400" : "bg-green-600 hover:bg-green-700"}
                      transition-all`}
        >
          {loading ? "Analyzing..." : "Detect Disease"}
        </button>
      </form>

      {/* Results */}
      {result && (
        <div className="mt-10 bg-white p-6 rounded-2xl shadow-lg w-full max-w-5xl">
          {result.disease === "Unknown / Not a leaf" ? (
            <div className="text-center text-red-500">
              <h2 className="text-xl font-bold mb-2">⚠️ Unrecognized Image</h2>
              <p className="mb-3 text-gray-600">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </p>
              <p className="text-sm">
                Please upload a clear, single leaf image with minimal background
                noise and good lighting.
              </p>

              <div className="mt-5 bg-green-50 rounded-xl p-4">
                <h3 className="text-green-700 font-semibold mb-2">
                  📸 Tips for Better Results:
                </h3>
                <ul className="text-gray-700 text-sm text-left list-disc ml-6">
                  <li>Use natural lighting (avoid shadows).</li>
                  <li>Focus on a single leaf without touching other leaves.</li>
                  <li>Remove blue/bright backgrounds (use soil or plain surface).</li>
                </ul>
              </div>
            </div>
          ) : (
            <div>
              <h2 className="text-2xl font-bold text-green-700 mb-2">
                🩺 Disease: {result.disease}
              </h2>
              <p className="text-gray-600 mb-4">
                Confidence:{" "}
                <span className="font-semibold text-green-700">
                  {(result.confidence * 100).toFixed(2)}%
                </span>
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-green-800 mb-2">
                    🌱 Recommended Solution
                  </h3>
                  <p className="text-gray-700 mb-4 leading-relaxed">
                    {result.solution}
                  </p>

                  <h3 className="text-lg font-semibold text-green-800 mb-2">
                    🌿 Leaf Characteristics
                  </h3>
                  <ul className="list-disc ml-6 text-gray-700 text-sm">
                    {Object.entries(result.leaf_characteristics).map(
                      ([k, v]) => (
                        <li key={k}>
                          <span className="font-semibold capitalize">{k}:</span>{" "}
                          {v}
                        </li>
                      )
                    )}
                  </ul>
                </div>

                {allProbs.length > 0 && (
                  <div>
                    <h3 className="text-lg font-semibold text-green-800 mb-2">
                      Confidence Distribution
                    </h3>
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart data={allProbs.slice(0, 6)}>
                        <XAxis dataKey="label" hide />
                        <Tooltip />
                        <Bar dataKey="value">
                          {allProbs
                            .slice(0, 6)
                            .map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={COLORS[index % COLORS.length]}
                              />
                            ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Upload Again */}
          <div className="mt-6 flex justify-end">
            <button
              onClick={() => {
                setResult(null);
                setFile(null);
                setImagePreview(null);
                setAllProbs([]);
              }}
              className="py-2 px-4 bg-green-500 hover:bg-green-600 text-white rounded-md font-medium transition-all"
            >
              Upload Another Image
            </button>
          </div>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="mt-10 bg-white shadow-md p-6 rounded-xl w-full max-w-3xl">
          <h3 className="text-green-800 text-lg font-semibold mb-3">
            📜 Recent Analyses
          </h3>
          <div className="space-y-2">
            {history.map((item, i) => (
              <div
                key={i}
                className="flex justify-between text-gray-700 text-sm border-b py-1"
              >
                <span>{item.name}</span>
                <span>{item.confidence}%</span>
                <span className="text-gray-400">{item.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <footer className="mt-10 text-gray-500 text-xs text-center">
        🌾 Built with FastAPI, PyTorch, React & Tailwind — v1.0
      </footer>
    </div>
  );
}
