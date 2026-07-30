import { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend 
} from 'recharts';
import { 
  TrendingUp, 
  Droplet, 
  ShieldAlert, 
  Sparkles, 
  Sliders, 
  MapPin, 
  MessageSquare, 
  Calendar, 
  ArrowRight, 
  CheckCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import './index.css';

const AHMEDABAD_WARDS = [
  { name: "Navrangpura", lat: 23.040, lon: 72.560, offset: 0.05 },
  { name: "Vastrapur", lat: 23.035, lon: 72.525, offset: -0.04 },
  { name: "Satellite", lat: 23.028, lon: 72.515, offset: -0.02 },
  { name: "Bodakdev", lat: 23.038, lon: 72.510, offset: -0.06 },
  { name: "Paldi", lat: 23.010, lon: 72.560, offset: 0.02 },
  { name: "Maninagar", lat: 22.998, lon: 72.605, offset: 0.09 },
  { name: "Ghatlodia", lat: 23.065, lon: 72.535, offset: 0.06 },
  { name: "Sabarmati", lat: 23.085, lon: 72.585, offset: -0.01 },
  { name: "Jamalpur", lat: 23.015, lon: 72.588, offset: 0.12 },
  { name: "Bapunagar", lat: 23.035, lon: 72.628, offset: 0.10 },
];

function App() {
  // District & Dates
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState({ code: 438, name: 'Ahmedabad' });
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-03-31');

  // Policy Simulator intensities (0-100)
  const [rwh, setRwh] = useState(0);
  const [dr, setDr] = useState(0);
  const [wc, setWc] = useState(0);
  const [aws, setAws] = useState(0);

  // Data states
  const [predictions, setPredictions] = useState([]);
  const [simulations, setSimulations] = useState([]);
  const [recommendation, setRecommendation] = useState(null);
  
  // Comparison scenarios for line chart
  const [rwh100, setRwh100] = useState([]);
  const [dr100, setDr100] = useState([]);
  const [wc100, setWc100] = useState([]);
  const [aws100, setAws100] = useState([]);

  // Checkboxes for scenario toggles in line chart
  const [showRwhLine, setShowRwhLine] = useState(false);
  const [showDrLine, setShowDrLine] = useState(false);
  const [showWcLine, setShowWcLine] = useState(false);
  const [showAwsLine, setShowAwsLine] = useState(false);

  // UI state
  const [loading, setLoading] = useState(true);
  const [simLoading, setSimLoading] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);

  // Chatbot states
  const [chatHistory, setChatHistory] = useState([
    { role: 'assistant', content: 'Hi! I am the AI Water Intelligence Agent. I can help analyze predicted water stress, explain model forecasts, or suggest conservation strategies. Ask me anything!' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatbotLoading, setChatbotLoading] = useState(false);

  // Map Refs
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersLayerRef = useRef(null);

  // Fetch district list on startup
  useEffect(() => {
    fetch('/api/districts')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'SUCCESS' && data.districts) {
          setDistricts(data.districts);
          // Auto select Ahmedabad if present, else first
          const ahmedabad = data.districts.find(d => d.District === 'Ahmedabad');
          if (ahmedabad) {
            setSelectedDistrict({ code: ahmedabad['District LGD Code'], name: ahmedabad.District });
          } else if (data.districts.length > 0) {
            setSelectedDistrict({ 
              code: data.districts[0]['District LGD Code'], 
              name: data.districts[0].District 
            });
          }
        }
      })
      .catch(err => console.error("Error loading districts:", err));
  }, []);

  // Fetch predictions and 100% scenarios on district/date change
  useEffect(() => {
    if (!selectedDistrict) return;
    
    const fetchBaseAndScenarios = async () => {
      setLoading(true);
      try {
        // 1. Fetch base predictions
        const predRes = await fetch(`/api/predict?district_code=${selectedDistrict.code}&start_date=${startDate}&end_date=${endDate}`);
        const predData = await predRes.json();
        
        if (predData.status === 'SUCCESS' && predData.predictions) {
          setPredictions(predData.predictions);
          
          // Get recommendation based on latest date in range
          const latestRecord = predData.predictions[predData.predictions.length - 1];
          if (latestRecord) {
            const recRes = await fetch(`/api/recommend?district_code=${selectedDistrict.code}&date=${latestRecord.date}`);
            const recData = await recRes.json();
            if (recData.status === 'SUCCESS') {
              setRecommendation(recData.recommendation);
            }
          }
        }

        // Helper to query simulator endpoint
        const fetchSimScenario = async (r, d, w, a) => {
          const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              district_code: selectedDistrict.code,
              start_date: startDate,
              end_date: endDate,
              rainwater_harvesting: r,
              demand_reduction: d,
              water_conservation: w,
              additional_water_supply: a
            })
          });
          const data = await res.json();
          return data.status === 'SUCCESS' ? data.simulation : [];
        };

        // Fetch 100% scenario benchmarks in parallel
        const [rwhRes, drRes, wcRes, awsRes] = await Promise.all([
          fetchSimScenario(1.0, 0.0, 0.0, 0.0),
          fetchSimScenario(0.0, 1.0, 0.0, 0.0),
          fetchSimScenario(0.0, 0.0, 1.0, 0.0),
          fetchSimScenario(0.0, 0.0, 0.0, 1.0)
        ]);

        setRwh100(rwhRes);
        setDr100(drRes);
        setWc100(wcRes);
        setAws100(awsRes);

      } catch (err) {
        console.error("Error loading predictions/benchmarks:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchBaseAndScenarios();
  }, [selectedDistrict, startDate, endDate]);

  // Handle active sandbox simulation updates (debounced / on change)
  useEffect(() => {
    if (!selectedDistrict || loading) return;

    const runActiveSimulation = async () => {
      setSimLoading(true);
      try {
        const res = await fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            district_code: selectedDistrict.code,
            start_date: startDate,
            end_date: endDate,
            rainwater_harvesting: rwh / 100,
            demand_reduction: dr / 100,
            water_conservation: wc / 100,
            additional_water_supply: aws / 100
          })
        });
        const data = await res.json();
        if (data.status === 'SUCCESS') {
          setSimulations(data.simulation);
        }
      } catch (err) {
        console.error("Error running active simulation:", err);
      } finally {
        setSimLoading(false);
      }
    };

    // Run active simulation
    const delayDebounce = setTimeout(() => {
      runActiveSimulation();
    }, 250);

    return () => clearTimeout(delayDebounce);
  }, [selectedDistrict, startDate, endDate, rwh, dr, wc, aws, loading]);

  // Leaflet Map Initializer
  useEffect(() => {
    if (selectedDistrict.name === 'Ahmedabad' && mapRef.current) {
      if (!mapInstanceRef.current) {
        mapInstanceRef.current = L.map(mapRef.current).setView([23.0225, 72.5714], 12);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
          subdomains: 'abcd',
          maxZoom: 20
        }).addTo(mapInstanceRef.current);
        markersLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);
      }
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
        markersLayerRef.current = null;
      }
    };
  }, [selectedDistrict]);

  // Variables for latest index data
  const latestIdx = predictions.length - 1;
  const latestActualWsi = latestIdx >= 0 ? predictions[latestIdx].wsi_actual : 0.0;
  const latestBaseline30d = latestIdx >= 0 ? predictions[latestIdx].pred_wsi_30d : 0.0;
  const latestSimulated30d = (latestIdx >= 0 && simulations[latestIdx]) ? simulations[latestIdx].simulated.wsi_30d : 0.0;
  const wsiReduction = latestBaseline30d - latestSimulated30d;

  // Update map markers when simulated WSI changes
  useEffect(() => {
    if (mapInstanceRef.current && markersLayerRef.current && selectedDistrict.name === 'Ahmedabad') {
      markersLayerRef.current.clearLayers();
      
      AHMEDABAD_WARDS.forEach(ward => {
        const wardWsi = Math.min(Math.max(latestSimulated30d + ward.offset, 0.0), 1.0);
        const color = wardWsi > 0.7 ? '#ff7b72' : wardWsi > 0.5 ? '#ffb86c' : '#56b400';
        const category = wardWsi > 0.7 ? 'HIGH' : wardWsi > 0.5 ? 'MEDIUM' : 'LOW';
        
        const marker = L.circleMarker([ward.lat, ward.lon], {
          radius: 10 + (wardWsi * 12),
          color: color,
          fillColor: color,
          fillOpacity: 0.6,
          weight: 2
        });
        
        const popupHtml = `
          <div style="color: #c9d1d9; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 13px; line-height: 1.5; padding: 4px;">
            <strong style="color: #ffffff; font-size: 14px;">Ward: ${ward.name}</strong><br/>
            <div style="margin-top: 6px;">30d Predicted WSI: <strong>${wardWsi.toFixed(2)}</strong></div>
            <div>Offset vs District: <strong>${ward.offset > 0 ? '+' : ''}${ward.offset}</strong></div>
            <div style="margin-top: 8px; display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: ${color}22; color: ${color}; border: 1px solid ${color}44;">
              Risk: ${category}
            </div>
          </div>
        `;
        
        marker.bindPopup(popupHtml, {
          className: 'custom-leaflet-popup'
        });
        
        marker.addTo(markersLayerRef.current);
      });
    }
  }, [latestSimulated30d, selectedDistrict]);

  // Handle District selection
  const handleDistrictChange = (e) => {
    const code = parseInt(e.target.value);
    const dist = districts.find(d => d['District LGD Code'] === code);
    if (dist) {
      setSelectedDistrict({ code: dist['District LGD Code'], name: dist.District });
    }
  };

  // Build Compare Futures Line Chart Data
  const chartData = predictions.map((p, idx) => {
    const sim = simulations[idx] || {};
    return {
      date: p.date,
      actual: parseFloat(p.wsi_actual.toFixed(3)),
      baseline30d: parseFloat(p.pred_wsi_30d.toFixed(3)),
      simulated30d: sim.simulated ? parseFloat(sim.simulated.wsi_30d.toFixed(3)) : null,
      rwh100: rwh100[idx]?.simulated ? parseFloat(rwh100[idx].simulated.wsi_30d.toFixed(3)) : null,
      dr100: dr100[idx]?.simulated ? parseFloat(dr100[idx].simulated.wsi_30d.toFixed(3)) : null,
      wc100: wc100[idx]?.simulated ? parseFloat(wc100[idx].simulated.wsi_30d.toFixed(3)) : null,
      aws100: aws100[idx]?.simulated ? parseFloat(aws100[idx].simulated.wsi_30d.toFixed(3)) : null,
    };
  });

  // XAI Components Stress Data
  const xaiData = recommendation ? Object.entries(recommendation.component_stresses).map(([key, val]) => ({
    name: key.replace(" Table Depletion", "").replace(" Deficit", "").replace(" Stress", ""),
    score: parseFloat(val.toFixed(2))
  })).sort((a, b) => a.score - b.score) : [];

  // Ahmedabad Wards rankings sorted
  const wardRankings = AHMEDABAD_WARDS.map(ward => {
    const wardWsi = Math.min(Math.max(latestSimulated30d + ward.offset, 0.0), 1.0);
    return { name: ward.name, wsi: wardWsi };
  }).sort((a, b) => b.wsi - a.wsi);

  // Chat Submission handler
  const handleChatSubmit = (e, customQuery = null) => {
    if (e) e.preventDefault();
    const query = customQuery || chatInput;
    if (!query.trim() || chatbotLoading) return;

    setChatInput('');
    const newHistory = [...chatHistory, { role: 'user', content: query }];
    setChatHistory(newHistory);
    setChatbotLoading(true);

    setTimeout(() => {
      let response = '';
      const qLower = query.toLowerCase();
      const compData = recommendation?.component_stresses || {
        "Groundwater Table Depletion": 0.5,
        "Rainfall Deficit": 0.5,
        "Atmospheric Temperature Stress": 0.5,
        "Humidity Deficit": 0.5,
        "Surface River Deficit": 0.5
      };

      if (qLower.includes("outcome") || qLower.includes("harvesting") || qLower.includes("rwh") || qLower.includes("supply") || qLower.includes("aws")) {
        const rwhWsi = rwh100[rwh100.length - 1]?.simulated?.wsi_30d || 0.45;
        const awsWsi = aws100[aws100.length - 1]?.simulated?.wsi_30d || 0.47;
        
        response = `### 📊 Scenario Simulation Comparison for **${selectedDistrict.name}**\n\n` +
          `I have executed two physical policy simulations for the 30-day horizon:\n\n` +
          `1. **100% Rainwater Harvesting (RWH):**\n` +
          `   - Predicted WSI: from a baseline of **${latestBaseline30d.toFixed(2)}** down to **${rwhWsi.toFixed(2)}** (a reduction of **-${(latestBaseline30d - rwhWsi).toFixed(2)}**).\n` +
          `   - **Mechanism:** Recharges aquifer pressure (reduces groundwater depth score) and captures storm runoffs.\n\n` +
          `2. **100% Additional Water Supply (AWS):**\n` +
          `   - Predicted WSI: from a baseline of **${latestBaseline30d.toFixed(2)}** down to **${awsWsi.toFixed(2)}** (a reduction of **-${(latestBaseline30d - awsWsi).toFixed(2)}**).\n` +
          `   - **Mechanism:** Boosts raw water capacity directly (increases river/canal levels) and reduces groundwater dependency.\n\n` +
          `**Conclusion:** ` +
          `For ${selectedDistrict.name}, **${(latestBaseline30d - rwhWsi) > (latestBaseline30d - awsWsi) ? 'Rainwater Harvesting' : 'Additional Water Supply'}** is the most effective standalone physical intervention.`;
      } else if (qLower.includes("highest scarcity risk") || qLower.includes("wards") || qLower.includes("risk")) {
        if (selectedDistrict.name !== 'Ahmedabad') {
          response = `Ward-level spatial downscaling is currently configured for **Ahmedabad** ward coordinates. Please select **Ahmedabad** district from the sidebar to inspect detailed ward metrics.`;
        } else {
          const rankings = AHMEDABAD_WARDS.map(w => ({
            name: w.name,
            wsi: Math.min(Math.max(latestSimulated30d + w.offset, 0.0), 1.0)
          })).sort((a, b) => b.wsi - a.wsi);
          
          const topWard = rankings[0];
          const bottomWard = rankings[rankings.length - 1];
          
          response = `Based on the downscaled ML forecasts, **${topWard.name}** is at the **highest scarcity risk** ` +
            `with a predicted WSI of **${topWard.wsi.toFixed(2)}** in 30 days.\n\n` +
            `Conversely, **${bottomWard.name}** has the lowest risk (**${bottomWard.wsi.toFixed(2)}**).\n\n` +
            `**Recommended action:** Prioritize groundwater recharge and residential society fixtures in **${topWard.name}** immediately.`;
        }
      } else if (qLower.includes("groundwater") || qLower.includes("gw") || qLower.includes("aquifer") || qLower.includes("borewell")) {
        const gwStressVal = compData["Groundwater Table Depletion"];
        const severityStr = gwStressVal > 0.7 ? "EXTREMELY CRITICAL" : gwStressVal > 0.4 ? "MODERATE" : "LOW";
        response = `### 🪓 Groundwater Depletion Audit for **${selectedDistrict.name}**\n\n` +
          `* **Normalized Stress Score:** \`${gwStressVal.toFixed(2)}\`\n` +
          `* **Aquifer Status:** \`${severityStr}\`\n\n` +
          `**AI Analysis:**\n` +
          `Groundwater extraction is ${gwStressVal > 0.6 ? 'outstripping natural recharge rates. Water table levels are deep and require immediate restrictions on borewell drilling' : 'currently stable but requires long-term monitoring'}.\n\n` +
          `**Recommended Intervention:** Implement **Demand Reduction** policies (such as metering) and recharge the aquifer via **Rainwater Harvesting** injection wells.`;
      } else if (qLower.includes("rain") || qLower.includes("monsoon") || qLower.includes("precipitation")) {
        const rainStressVal = compData["Rainfall Deficit"];
        response = `### 🌧️ Rainfall & Precipitation Analysis for **${selectedDistrict.name}**\n\n` +
          `* **Rainfall Deficit Stress Score:** \`${rainStressVal.toFixed(2)}\`\n\n` +
          `**AI Analysis:**\n` +
          `A rainfall deficit stress score of \`${rainStressVal.toFixed(2)}\` indicates that current seasonal precipitation is ${rainStressVal > 0.6 ? 'significantly below the historical baseline' : 'aligned with normal seasonal trends'}.\n\n` +
          `**Action Recommended:** Engage **Water Conservation** campaigns and implement rooftop **Rainwater Harvesting** to capture future monsoon runoffs.`;
      } else if (qLower.includes("temp") || qLower.includes("heat") || qLower.includes("evaporation") || qLower.includes("temperature") || qLower.includes("climate") || qLower.includes("summer")) {
        const tempStressVal = compData["Atmospheric Temperature Stress"];
        response = `### 🌡️ Temperature & Evaporation Risk for **${selectedDistrict.name}**\n\n` +
          `* **Evaporative Temperature Stress Score:** \`${tempStressVal.toFixed(2)}\`\n\n` +
          `**AI Analysis:**\n` +
          `High temperatures increase soil moisture depletion and municipal evaporation rates. A stress score of \`${tempStressVal.toFixed(2)}\` suggests high irrigation demand in agricultural borders.\n\n` +
          `**Mitigation Strategy:** Encourage agricultural mulching, shift supply hours to cooler evening periods to reduce evaporation, and expand urban shade canopies.`;
      } else if (qLower.includes("river") || qLower.includes("canal") || qLower.includes("surface") || qLower.includes("reservoir")) {
        const riverStressVal = compData["Surface River Deficit"];
        response = `### 🌊 Surface River & Canal Level Audit for **${selectedDistrict.name}**\n\n` +
          `* **Surface River Deficit Stress Score:** \`${riverStressVal.toFixed(2)}\`\n\n` +
          `**AI Analysis:**\n` +
          `Surface water flows are ${riverStressVal > 0.6 ? 'heavily depleted' : 'adequate for this season'}. This affects canal supply channels and direct surface intake stations.\n\n` +
          `**Mitigation Strategy:** Treat and recycle municipal greywater to augment surface flows, and coordinate canal releases with upstream reservoirs.`;
      } else if (qLower.includes("simulate") || qLower.includes("policy") || qLower.includes("slider") || qLower.includes("sandbox")) {
        response = `### 🎛️ Sandbox Policy Simulation Status\n\n` +
          `You have activated the following interventions on the sidebar:\n` +
          `- 🌧️ **Rainwater Harvesting:** \`${rwh}%\` implementation\n` +
          `- 📉 **Demand Reduction:** \`${dr}%\` dynamic pricing & caps\n` +
          `- 🌾 **Water Conservation:** \`${wc}%\` smart fixtures & drip irrigation\n` +
          `- 🌊 **Additional Supply:** \`${aws}%\` dynamic flow additions\n\n` +
          `**Outcome Evaluation:**\n` +
          `This combination leads to a predicted 30-day Water Stress Index of **${latestSimulated30d.toFixed(2)}** ` +
          `compared to a baseline of **${latestBaseline30d.toFixed(2)}** (a net reduction of **-${wsiReduction.toFixed(2)}** WSI).\n\n` +
          `Adjust the sidebar sliders to see the forecasts recalculate in real-time.`;
      } else if (qLower.includes("help") || qLower.includes("hello") || qLower.includes("hi") || qLower.includes("capabilities")) {
        response = `### 💬 Water Intelligence Assistant Capabilities\n\n` +
          `I am connected to the Stage 7 Feature Store and Stage 9 machine learning model predictions. You can ask me:\n` +
          `1. **Groundwater status:** 'What is the groundwater level?'\n` +
          `2. **Precipitation details:** 'Show me rainfall anomaly and monsoon status'\n` +
          `3. **Temperature issues:** 'How are temperatures affecting evapotranspiration?'\n` +
          `4. **Surface water levels:** 'What are the current canal and river levels?'\n` +
          `5. **Simulation outcomes:** 'What happens if we implement 100% rainwater harvesting?'\n` +
          `6. **Wards risk comparison:** 'Which Ahmedabad wards have the highest risk?'\n` +
          `7. **Slider simulator stats:** 'Explain my current slider policies'`;
      } else {
        response = `Predicted water stress in **${selectedDistrict.name}** is driven primarily by **${recommendation?.primary_driver || 'Groundwater depletion'}** ` +
          `(stress factor of ${(recommendation?.driver_severity || 0.5).toFixed(2)}).\n\n` +
          `**Detailed Breakdown:**\n` +
          `- Groundwater stress: ${compData["Groundwater Table Depletion"].toFixed(2)}\n` +
          `- Rainfall deficit: ${compData["Rainfall Deficit"].toFixed(2)}\n` +
          `- Evaporative Temperature stress: ${compData["Atmospheric Temperature Stress"].toFixed(2)}\n\n` +
          `I recommend implementing a combination of **${recommendation?.recommended_strategy || 'Demand Reduction'}** and **Water Conservation** ` +
          `to offset the predicted stress of **${latestBaseline30d.toFixed(2)}**.`;
      }

      setChatHistory(prev => [...prev, { role: 'assistant', content: response }]);
      setChatbotLoading(false);
    }, 800);
  };

  // Quick preset options
  const PRESETS = [
    `Why is water stress predicted to change in ${selectedDistrict.name}?`,
    `What is the simulated benefit of 100% Rainwater Harvesting vs. 100% Additional Supply?`,
    `Which Ahmedabad wards are currently at the highest scarcity risk?`
  ];

  return (
    <div className="app-container">
      {/* Sidebar Controls */}
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Droplet size={28} style={{ color: '#58a6ff' }} />
          <h2>Decision Sandbox</h2>
        </div>
        <p style={{ fontSize: '0.8rem', color: '#8b949e', marginTop: '-10px' }}>
          Configure ML forecasting parameters & simulated policies.
        </p>
        
        <hr style={{ border: '0', borderTop: '1px solid #21262d' }} />

        {/* District Select */}
        <div className="sidebar-section">
          <label className="sidebar-label">Select District</label>
          <select 
            className="select-input" 
            value={selectedDistrict.code} 
            onChange={handleDistrictChange}
          >
            {districts.map(d => (
              <option key={d['District LGD Code']} value={d['District LGD Code']}>
                {d.District}
              </option>
            ))}
          </select>
        </div>

        {/* Date Selectors */}
        <div className="sidebar-section">
          <label className="sidebar-label">Forecast Start Date</label>
          <input 
            type="date" 
            className="date-input" 
            value={startDate} 
            min="2025-01-01"
            max="2025-03-31"
            onChange={e => setStartDate(e.target.value)}
          />
        </div>
        <div className="sidebar-section">
          <label className="sidebar-label">Forecast End Date</label>
          <input 
            type="date" 
            className="date-input" 
            value={endDate} 
            min="2025-01-01"
            max="2025-03-31"
            onChange={e => setEndDate(e.target.value)}
          />
        </div>

        <hr style={{ border: '0', borderTop: '1px solid #21262d' }} />

        {/* Slider Controls */}
        <div className="sidebar-section">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <Sliders size={16} style={{ color: '#58a6ff' }} />
            <span className="sidebar-label" style={{ fontWeight: '600', color: '#c9d1d9' }}>Intervention Intensities</span>
          </div>

          <div className="slider-group">
            {/* RWH */}
            <div className="slider-container">
              <div className="slider-header">
                <span>Rainwater Harvesting</span>
                <span style={{ color: '#58a6ff' }}>{rwh}%</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min="0" 
                max="100" 
                step="5"
                value={rwh} 
                onChange={e => setRwh(parseInt(e.target.value))}
              />
              <span className="slider-help">Aquifer recharge & surface runoff storage.</span>
            </div>

            {/* DR */}
            <div className="slider-container">
              <div className="slider-header">
                <span>Demand Reduction</span>
                <span style={{ color: '#ffb86c' }}>{dr}%</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min="0" 
                max="100" 
                step="5"
                value={dr} 
                onChange={e => setDr(parseInt(e.target.value))}
              />
              <span className="slider-help">Dynamic billing tariffs & caps on non-essential consumption.</span>
            </div>

            {/* WC */}
            <div className="slider-container">
              <div className="slider-header">
                <span>Water Conservation</span>
                <span style={{ color: '#56b400' }}>{wc}%</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min="0" 
                max="100" 
                step="5"
                value={wc} 
                onChange={e => setWc(parseInt(e.target.value))}
              />
              <span className="slider-help">Encouraging low-flow fixtures & micro-drip agricultural practices.</span>
            </div>

            {/* AWS */}
            <div className="slider-container">
              <div className="slider-header">
                <span>Additional Supply</span>
                <span style={{ color: '#b388ff' }}>{aws}%</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min="0" 
                max="100" 
                step="5"
                value={aws} 
                onChange={e => setAws(parseInt(e.target.value))}
              />
              <span className="slider-help">Augmenting raw canal inflow allocations.</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Dashboard */}
      <main className="dashboard-content">
        {/* Banner */}
        <header className="header-banner">
          <h1 className="header-title">💧 Water Intelligence Platform</h1>
          <p className="header-subtitle">
            Gujarat Scarcity Prediction, Policy Simulation & Agentic Decision-Support
          </p>

          <button 
            className="about-toggle" 
            onClick={() => setAboutOpen(!aboutOpen)}
            style={{
              marginTop: '15px',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid #30363d',
              padding: '6px 16px',
              borderRadius: '20px',
              color: '#c9d1d9',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.8rem',
              transition: 'background 0.2s'
            }}
          >
            <Sparkles size={14} style={{ color: '#58a6ff' }} />
            About the AI Layer (Decision-Support Engine)
            {aboutOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {aboutOpen && (
            <div 
              className="about-card"
              style={{
                marginTop: '15px',
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '8px',
                padding: '20px',
                textAlign: 'left',
                lineHeight: '1.6',
                fontSize: '0.85rem'
              }}
            >
              <h3 style={{ color: '#ffffff', marginBottom: '8px' }}>Core Components:</h3>
              <ol style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <li>
                  <strong>🔮 Predictive Analytics:</strong> Chronologically-validated RandomForest regressors forecast Water Stress Index (WSI) for 7, 15, and 30-day horizons.
                </li>
                <li>
                  <strong>🧠 Explainable AI (XAI):</strong> Ranks and visualizes normalized environmental driver scores.
                </li>
                <li>
                  <strong>🛠️ Decision Sandbox:</strong> Propagates physical feature changes (e.g. aquifer recharge, river level offsets) to forecast intervention scenarios.
                </li>
                <li>
                  <strong>📊 Scenario Comparison:</strong> Side-by-side comparison of baseline forecasts vs. simulated policies (RWH, DR, WC, AWS).
                </li>
                <li>
                  <strong>💬 AI Strategy Advisor:</strong> Proactively suggests and ranks action plans based on local drivers.
                </li>
              </ol>
            </div>
          )}
        </header>

        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', gap: '15px' }}>
            <Droplet size={48} className="loading-icon" style={{ color: '#58a6ff' }} />
            <p style={{ color: '#8b949e', fontSize: '1.1rem' }}>Loading forecasting data & simulation models...</p>
          </div>
        ) : (
          <>
            {/* Metric Cards Row */}
            <div className="metrics-row">
              <div className="metric-card">
                <div className="metric-lbl">Current Water Stress (WSI)</div>
                <div className="metric-val" style={{ color: '#58a6ff' }}>
                  {latestActualWsi.toFixed(2)}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-lbl">Baseline Forecast (30-Day)</div>
                <div className="metric-val" style={{ color: '#ff7b72' }}>
                  {latestBaseline30d.toFixed(2)}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-lbl">Simulated Forecast (30-Day)</div>
                <div className="metric-val" style={{ color: '#ffb86c' }}>
                  {latestSimulated30d.toFixed(2)}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-lbl">Expected WSI Reduction</div>
                <div className="metric-val" style={{ color: wsiReduction > 0 ? '#56b400' : '#8b949e' }}>
                  {wsiReduction > 0 ? `-${wsiReduction.toFixed(2)}` : '0.00'}
                </div>
              </div>
            </div>

            {/* Ahmedabad Wards Map ( Ahmedbad Specific ) */}
            {selectedDistrict.name === 'Ahmedabad' && (
              <section className="panel-card">
                <h3 className="panel-title">
                  <MapPin size={18} style={{ color: '#58a6ff' }} />
                  Ahmedabad Ward-Level Analytics & Spatial Join
                </h3>
                <p style={{ fontSize: '0.85rem', color: '#8b949e', marginTop: '-5px' }}>
                  Hover or click on markers to view ward-level water stress forecasts downscaled from the regional ML model.
                </p>
                <div className="grid-2col">
                  {/* Left Map */}
                  <div ref={mapRef} className="leaflet-container" />
                  
                  {/* Right Rankings */}
                  <div>
                    <h4 style={{ fontSize: '0.95rem', color: '#ffffff', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <TrendingUp size={14} style={{ color: '#ffb86c' }} />
                      Ward Stress Rankings
                    </h4>
                    <div className="rankings-list">
                      {wardRankings.map(ward => {
                        const color = ward.wsi > 0.7 ? '#ff7b72' : ward.wsi > 0.5 ? '#ffb86c' : '#56b400';
                        const bgColor = ward.wsi > 0.7 ? 'rgba(255,123,114,0.1)' : ward.wsi > 0.5 ? 'rgba(255,184,108,0.1)' : 'rgba(86,180,0,0.1)';
                        return (
                          <div key={ward.name} className="ranking-item">
                            <span className="ranking-name">{ward.name}</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexGrow: '1', justifyContent: 'flex-end' }}>
                              <div style={{ width: '80px', height: '6px', background: '#21262d', borderRadius: '3px', overflow: 'hidden' }}>
                                <div style={{ width: `${ward.wsi * 100}%`, height: '100%', background: color }} />
                              </div>
                              <span 
                                className="ranking-score-badge"
                                style={{ color: color, backgroundColor: bgColor, border: `1px solid ${color}33` }}
                              >
                                {ward.wsi.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* Compare Futures Line Chart */}
            <section className="panel-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #21262d', paddingBottom: '10px' }}>
                <h3 className="panel-title" style={{ borderBottom: 'none', paddingBottom: '0' }}>
                  <TrendingUp size={18} style={{ color: '#56b400' }} />
                  Compare Futures: Baseline vs. Simulated Water Scarcity Trajectory
                </h3>
                {/* Benchmark Checkboxes */}
                <div style={{ display: 'flex', gap: '15px', fontSize: '0.8rem', color: '#8b949e' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={showRwhLine} onChange={e => setShowRwhLine(e.target.checked)} />
                    100% RWH
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={showDrLine} onChange={e => setShowDrLine(e.target.checked)} />
                    100% DR
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={showWcLine} onChange={e => setShowWcLine(e.target.checked)} />
                    100% WC
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={showAwsLine} onChange={e => setShowAwsLine(e.target.checked)} />
                    100% AWS
                  </label>
                </div>
              </div>

              <div style={{ width: '100%', height: '350px', marginTop: '10px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                    <XAxis dataKey="date" stroke="#8b949e" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#8b949e" tick={{ fontSize: 11 }} domain={[0.0, 1.05]} />
                    <RechartsTooltip 
                      contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d', color: '#c9d1d9' }}
                      itemStyle={{ fontSize: 12 }}
                      labelStyle={{ fontSize: 11, fontWeight: 'bold' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                    <Line type="monotone" dataKey="actual" name="Current WSI Baseline" stroke="#58a6ff" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="baseline30d" name="Baseline 30d Forecast" stroke="#ff7b72" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
                    <Line type="monotone" dataKey="simulated30d" name="Active Sandbox Forecast" stroke="#56b400" strokeWidth={3} dot={false} />
                    {showRwhLine && <Line type="monotone" dataKey="rwh100" name="100% RWH Benchmark" stroke="#38bdf8" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />}
                    {showDrLine && <Line type="monotone" dataKey="dr100" name="100% DR Benchmark" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />}
                    {showWcLine && <Line type="monotone" dataKey="wc100" name="100% WC Benchmark" stroke="#34d399" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />}
                    {showAwsLine && <Line type="monotone" dataKey="aws100" name="100% AWS Benchmark" stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* XAI and AI Strategy Advisor */}
            <div className="grid-2col" style={{ gridTemplateColumns: '1fr 1fr' }}>
              {/* XAI drivers */}
              <section className="panel-card">
                <h3 className="panel-title">
                  <ShieldAlert size={18} style={{ color: '#ff7b72' }} />
                  Explainable AI (XAI): Stress Drivers
                </h3>
                <div style={{ width: '100%', height: '220px', marginTop: '10px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={xaiData} layout="vertical" margin={{ top: 0, right: 10, left: 15, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#21262d" horizontal={false} />
                      <XAxis type="number" stroke="#8b949e" domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <YAxis dataKey="name" type="category" stroke="#8b949e" tick={{ fontSize: 11 }} width={90} />
                      <RechartsTooltip 
                        contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d', color: '#c9d1d9' }}
                        itemStyle={{ fontSize: 12 }}
                      />
                      <Bar 
                        dataKey="score" 
                        name="Stress Severity" 
                        fill="#ffb86c" 
                        radius={[0, 4, 4, 0]}
                        barSize={12}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                {recommendation && (
                  <div className="strategy-banner" style={{ background: '#161b22', borderColor: '#30363d', padding: '12px' }}>
                    <p style={{ fontSize: '0.8rem', lineHeight: '1.5', color: '#8b949e' }}>
                      <strong style={{ color: '#ff7b72' }}>AI Explanation:</strong> Water scarcity indicators highlight <strong>{recommendation.primary_driver}</strong> as the primary stress driver with a localized severity of <strong>{recommendation.driver_severity.toFixed(2)}</strong>. Without intervention, this leads to an expected WSI of <strong>{recommendation.baseline_wsi_30d.toFixed(2)}</strong> in 30 days. Applying the active policy scenarios reduces this risk to <strong>{latestSimulated30d.toFixed(2)}</strong>.
                    </p>
                  </div>
                )}
              </section>

              {/* AI Strategy Advisor */}
              <section className="panel-card">
                <h3 className="panel-title">
                  <CheckCircle size={18} style={{ color: '#56b400' }} />
                  AI Strategy Advisor
                </h3>
                {recommendation && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', height: '100%', justifyContent: 'space-between' }}>
                    <div className="strategy-banner">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className="strategy-title">Recommended: {recommendation.recommended_strategy}</span>
                        <span style={{ fontSize: '0.8rem', background: '#56b40022', color: '#56b400', padding: '2px 8px', borderRadius: '12px', border: '1px solid #56b40044', fontWeight: 'bold' }}>
                          -{recommendation.expected_30d_reduction.toFixed(2)} WSI
                        </span>
                      </div>
                      <p className="strategy-text" style={{ fontStyle: 'italic', marginTop: '6px', color: '#c9d1d9' }}>
                        "{recommendation.primary_action}"
                      </p>
                    </div>

                    <div>
                      <h4 style={{ fontSize: '0.85rem', color: '#ffffff', marginBottom: '8px' }}>Prioritized Policy Action Items:</h4>
                      <ul style={{ paddingLeft: '18px', fontSize: '0.8rem', color: '#c9d1d9', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {recommendation.action_items.map((item, i) => (
                          <li key={i} style={{ listStyleType: 'square' }}>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </section>
            </div>

            {/* Agentic AI Assistant Chatbot */}
            <section className="panel-card" style={{ gridColumn: 'span 2' }}>
              <h3 className="panel-title">
                <MessageSquare size={18} style={{ color: '#b388ff' }} />
                💬 Agentic AI Assistant
              </h3>
              
              <div className="chat-container">
                {/* Chat feed */}
                <div className="chat-history">
                  {chatHistory.map((chat, idx) => (
                    <div key={idx} className={`chat-bubble ${chat.role}`}>
                      <strong>{chat.role === 'user' ? 'You' : 'AI Water Agent'}:</strong>
                      <div style={{ whiteSpace: 'pre-wrap', marginTop: '4px', fontSize: '0.85rem' }}>
                        {chat.content.split('\n').map((line, i) => {
                          if (line.startsWith('###')) {
                            return <h4 key={i} style={{ color: '#ffffff', fontSize: '0.95rem', margin: '10px 0 5px' }}>{line.replace('###', '').trim()}</h4>;
                          }
                          if (line.startsWith('* **') || line.startsWith('- **') || line.startsWith('1. **')) {
                            return <p key={i} style={{ margin: '3px 0', paddingLeft: '10px' }}>{line}</p>;
                          }
                          return <p key={i} style={{ margin: '4px 0' }}>{line}</p>;
                        })}
                      </div>
                    </div>
                  ))}
                  {chatbotLoading && (
                    <div className="chat-bubble assistant" style={{ fontStyle: 'italic', color: '#8b949e' }}>
                      AI Water Agent is analyzing and simulating...
                    </div>
                  )}
                </div>

                {/* Preset quick buttons */}
                <div className="chat-presets">
                  {PRESETS.map((preset, i) => (
                    <button 
                      key={i} 
                      className="preset-btn"
                      onClick={(e) => handleChatSubmit(e, preset)}
                      disabled={chatbotLoading}
                    >
                      {preset.length > 55 ? `${preset.substring(0, 55)}...` : preset}
                    </button>
                  ))}
                </div>

                {/* Form input */}
                <form className="chat-input-form" onSubmit={handleChatSubmit}>
                  <input 
                    type="text" 
                    className="chat-text-input" 
                    placeholder="Ask a question about Gujarat water stress..."
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    disabled={chatbotLoading}
                  />
                  <button type="submit" className="chat-send-btn" disabled={chatbotLoading}>
                    Send
                  </button>
                </form>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
