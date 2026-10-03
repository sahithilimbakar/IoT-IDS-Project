"""
LIVE IoT INTRUSION DETECTION SYSTEM
REAL TRAFFIC ONLY
-----------------
- Captures packets from the selected real network interface.
- Does NOT generate fake/synthetic traffic.
- Does NOT read the training CSV during monitoring.
- Uses the existing trained reduced Random Forest model.
- Builds live packet-window features from captured traffic.
"""
import json
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scapy.all import IP, TCP, UDP, sniff, get_if_list
# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "local_environment_model.joblib"
)
# ============================================================
# SETTINGS
# ============================================================
# CICIoT2023 uses packet windows.
# We use a 10-packet live window.
PACKET_WINDOW_SIZE = 100
MAX_RESULTS = 100
REFRESH_SECONDS = 1
FEATURES = [
    "Number",
    "ack_flag_number",
    "HTTPS",
    "Tot size",
    "Rate",
]
# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(
    page_title="IoT IDS - Live Monitor",
    page_icon="🛡️",
    layout="wide",
)
# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 25px;
        font-weight: 600;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# WINDOWS NETWORK INTERFACES
# ============================================================
def get_windows_adapters():
    """
    Get Windows network adapters and their GUIDs.
    """
    adapters = {}
    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            """
            Get-NetAdapter |
            Select-Object Name, InterfaceGuid, Status, InterfaceDescription |
            ConvertTo-Json -Compress
            """,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return adapters
        output = result.stdout.strip()
        if not output:
            return adapters
        data = json.loads(output)
        if isinstance(data, dict):
            data = [data]
        for adapter in data:
            guid = str(
                adapter.get(
                    "InterfaceGuid",
                    ""
                )
            ).strip()
            if not guid:
                continue
            guid = guid.strip("{}").lower()
            adapters[guid] = {
                "name": str(
                    adapter.get(
                        "Name",
                        ""
                    )
                ).strip(),
                "status": str(
                    adapter.get(
                        "Status",
                        ""
                    )
                ).strip(),
                "description": str(
                    adapter.get(
                        "InterfaceDescription",
                        ""
                    )
                ).strip(),
            }
    except Exception:
        pass
    return adapters
def get_friendly_interfaces():
    """
    Match Windows adapters with Npcap interfaces.
    User sees:
        📶 Wi-Fi
        🔌 Ethernet
        🔄 Loopback
    Scapy receives:
        \\Device\\NPF_{GUID}
    """
    interfaces = {}
    windows_adapters = (
        get_windows_adapters()
    )
    try:
        npcap_interfaces = get_if_list()
    except Exception:
        return interfaces
    for npf_interface in npcap_interfaces:
        match = re.search(
            r"\{([0-9A-Fa-f-]{36})\}",
            npf_interface,
        )
        if not match:
            continue
        guid = (
            match.group(1)
            .lower()
        )
        adapter = windows_adapters.get(
            guid
        )
        if not adapter:
            continue
        name = adapter["name"]
        if not name:
            name = adapter["description"]
        if not name:
            continue
        lower_name = name.lower()
        if (
            "wi-fi" in lower_name
            or "wifi" in lower_name
        ):
            friendly_name = "📶 Wi-Fi"
        elif "ethernet" in lower_name:
            friendly_name = "🔌 Ethernet"
        elif "bluetooth" in lower_name:
            friendly_name = "🔵 Bluetooth"
        elif "vpn" in lower_name:
            friendly_name = "🌐 VPN"
        elif (
            "virtual" in lower_name
            or "hyper-v" in lower_name
            or "vmware" in lower_name
            or "virtualbox" in lower_name
        ):
            friendly_name = "🖥️ Virtual Adapter"
        else:
            friendly_name = (
                f"🖥️ {name}"
            )
        if (
            adapter["status"]
            .lower()
            == "up"
        ):
            friendly_name += (
                " (Connected)"
            )
        else:
            friendly_name += (
                " (Disconnected)"
            )
        interfaces[
            friendly_name
        ] = npf_interface
    # Loopback
    for npf_interface in npcap_interfaces:
        if "NPF_Loopback" in npf_interface:
            interfaces[
                "🔄 Loopback"
            ] = npf_interface
    return interfaces
# ============================================================
# LOAD EXISTING TRAINED MODEL
# ============================================================
@st.cache_resource
def load_ids_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model was not found:\n\n"
            f"{MODEL_PATH}"
        )
    bundle = joblib.load(
        MODEL_PATH
    )
    model = None
    features = None
    medians = {}
    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if isinstance(
        bundle,
        dict
    ):
        if "model" in bundle:
            model = bundle["model"]
        elif "estimator" in bundle:
            model = bundle["estimator"]
        elif "classifier" in bundle:
            model = bundle["classifier"]
        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------
        if "features" in bundle:
            features = bundle["features"]
        elif (
            "selected_features"
            in bundle
        ):
            features = (
                bundle[
                    "selected_features"
                ]
            )
        elif (
            "feature_names"
            in bundle
        ):
            features = (
                bundle[
                    "feature_names"
                ]
            )
        # ----------------------------------------------------
        # Medians
        # ----------------------------------------------------
        if "medians" in bundle:
            medians = bundle["medians"]
        elif (
            "feature_medians"
            in bundle
        ):
            medians = (
                bundle[
                    "feature_medians"
                ]
            )
    else:
        model = bundle
    if model is None:
        raise ValueError(
            "Could not find the trained "
            "model in ids_detector_reduced.joblib"
        )
    if features is None:
        features = FEATURES
    if isinstance(
        features,
        pd.Series
    ):
        features = features.tolist()
    elif isinstance(
        features,
        np.ndarray
    ):
        features = features.tolist()
    else:
        features = list(
            features
        )
    # --------------------------------------------------------
    # Safely handle medians
    # --------------------------------------------------------
    if isinstance(
        medians,
        pd.Series
    ):
        medians = (
            medians.to_dict()
        )
    elif isinstance(
        medians,
        pd.DataFrame
    ):
        if len(medians) > 0:
            medians = (
                medians
                .iloc[0]
                .to_dict()
            )
        else:
            medians = {}
    elif medians is None:
        medians = {}
    elif not isinstance(
        medians,
        dict
    ):
        try:
            medians = dict(
                medians
            )
        except Exception:
            medians = {}
    return (
        model,
        features,
        medians,
    )
# ============================================================
# LIVE PACKET WINDOW
# ============================================================
class PacketWindow:
    def __init__(
        self,
        first_time,
    ):
        self.first_time = (
            first_time
        )
        self.last_time = (
            first_time
        )
        self.packet_count = 0
        self.packet_sizes = []
        self.ack_flags = []
        self.https_flags = []
        self.src_ip = ""
        self.dst_ip = ""
        self.src_port = 0
        self.dst_port = 0
        self.protocol = "UNKNOWN"
    # --------------------------------------------------------
    # Add packet
    # --------------------------------------------------------
    def add_packet(
        self,
        packet
    ):
        current_time = time.time()
        self.last_time = (
            current_time
        )
        self.packet_count += 1
        # ----------------------------------------------------
        # IP information
        # ----------------------------------------------------
        if IP in packet:
            ip = packet[IP]
            self.src_ip = ip.src
            self.dst_ip = ip.dst
        # ----------------------------------------------------
        # TCP
        # ----------------------------------------------------
        if TCP in packet:
            self.protocol = "TCP"
            self.src_port = int(
                packet[TCP].sport
            )
            self.dst_port = int(
                packet[TCP].dport
            )
            # ACK flag
            try:
                flags = int(
                    packet[TCP].flags
                )
                ack = (
                    1
                    if flags & 0x10
                    else 0
                )
            except Exception:
                ack = 0
            self.ack_flags.append(
                ack
            )
            # HTTPS
            https = (
                1
                if (
                    int(packet[TCP].sport)
                    == 443
                    or
                    int(packet[TCP].dport)
                    == 443
                )
                else 0
            )
            self.https_flags.append(
                https
            )
        # ----------------------------------------------------
        # UDP
        # ----------------------------------------------------
        elif UDP in packet:
            self.protocol = "UDP"
            self.src_port = int(
                packet[UDP].sport
            )
            self.dst_port = int(
                packet[UDP].dport
            )
            self.ack_flags.append(
                0
            )
            # UDP/443 can be QUIC
            https = (
                1
                if (
                    int(packet[UDP].sport)
                    == 443
                    or
                    int(packet[UDP].dport)
                    == 443
                )
                else 0
            )
            self.https_flags.append(
                https
            )
        else:
            self.ack_flags.append(
                0
            )
            self.https_flags.append(
                0
            )
        # ----------------------------------------------------
        # Packet length
        #
        # CICIoT2023 describes Tot size as packet length.
        # We therefore store each packet length and later
        # calculate the window mean.
        # ----------------------------------------------------
        try:
            if IP in packet:
                packet_length = int(
                    packet[IP].len
                )
            else:
                packet_length = len(
                    packet
                )
        except Exception:
            packet_length = len(
                packet
            )
        self.packet_sizes.append(
            packet_length
        )
    # --------------------------------------------------------
    # Convert to model features
    # --------------------------------------------------------
    def to_features(self):
        duration = max(
            self.last_time
            - self.first_time,
            0.001,
        )
        packet_count = (
            self.packet_count
        )
        # Number of packets in window
        number = float(
            packet_count
        )
        # Mean ACK flag value
        ack_flag_number = float(
            np.mean(
                self.ack_flags
            )
            if self.ack_flags
            else 0
        )
        # Mean HTTPS indicator
        https = float(
            np.mean(
                self.https_flags
            )
            if self.https_flags
            else 0
        )
        # Mean packet length
        tot_size = float(
            np.mean(
                self.packet_sizes
            )
            if self.packet_sizes
            else 0
        )
        # Packet transmission rate
        rate = float(
            packet_count
            / duration
        )
        return {
            "Number":
                number,
            "ack_flag_number":
                ack_flag_number,
            "HTTPS":
                https,
            "Tot size":
                tot_size,
            "Rate":
                rate,
        }
# ============================================================
# CAPTURE MANAGER
# ============================================================
class CaptureManager:
    def __init__(self):
        self.running = False
        self.interface = None
        self.capture_thread = None
        self.lock = (
            threading.Lock()
        )
        self.current_windows = {}
        self.results = deque(
            maxlen=MAX_RESULTS
        )
        self.packet_count = 0
        self.flow_count = 0
        self.normal_count = 0
        self.attack_count = 0
        self.error = None
        self.started_at = None
    # --------------------------------------------------------
    # Flow key
    # --------------------------------------------------------
    @staticmethod
    def make_flow_key(
        src_ip,
        src_port,
        dst_ip,
        dst_port,
        protocol,
    ):
        endpoint1 = (
            src_ip,
            src_port,
        )
        endpoint2 = (
            dst_ip,
            dst_port,
        )
        if endpoint1 <= endpoint2:
            return (
                endpoint1,
                endpoint2,
                protocol,
            )
        return (
            endpoint2,
            endpoint1,
            protocol,
        )
    # --------------------------------------------------------
    # Process packet
    # --------------------------------------------------------
    def process_packet(
        self,
        packet
    ):
        if not self.running:
            return
        if IP not in packet:
            return
        ip = packet[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        # ----------------------------------------------------
        # TCP
        # ----------------------------------------------------
        if TCP in packet:
            protocol = "TCP"
            src_port = int(
                packet[TCP].sport
            )
            dst_port = int(
                packet[TCP].dport
            )
        # ----------------------------------------------------
        # UDP
        # ----------------------------------------------------
        elif UDP in packet:
            protocol = "UDP"
            src_port = int(
                packet[UDP].sport
            )
            dst_port = int(
                packet[UDP].dport
            )
        else:
            protocol = str(
                ip.proto
            )
            src_port = 0
            dst_port = 0
        key = self.make_flow_key(
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol,
        )
        with self.lock:
            self.packet_count += 1
            # ------------------------------------------------
            # Create window
            # ------------------------------------------------
            if (
                key
                not in self.current_windows
            ):
                self.current_windows[
                    key
                ] = PacketWindow(
                    time.time()
                )
            window = (
                self.current_windows[
                    key
                ]
            )
            window.add_packet(
                packet
            )
            # ------------------------------------------------
            # Analyze every 10 packets
            # ------------------------------------------------
            if (
                window.packet_count
                >= PACKET_WINDOW_SIZE
            ):
                completed_window = (
                    window
                )
                # Start a new window for
                # the same flow.
                self.current_windows[
                    key
                ] = PacketWindow(
                    time.time()
                )
                # Analyze outside lock
                # after this block.
                analyze = True
            else:
                completed_window = None
                analyze = False
        if analyze:
            self.analyze_window(
                completed_window
            )
    # --------------------------------------------------------
    # Analyze window
    # --------------------------------------------------------
    def analyze_window(
        self,
        window
    ):
        try:
            model, model_features, medians = (
                load_ids_model()
            )
            live_features = (
                window.to_features()
            )
            row = {}
            for feature in model_features:
                if feature in live_features:
                    row[feature] = (
                        live_features[
                            feature
                        ]
                    )
                else:
                    row[feature] = 0.0
            df = pd.DataFrame(
                [row]
            )
            # ------------------------------------------------
            # Apply saved medians
            # ------------------------------------------------
            if isinstance(
                medians,
                dict
            ):
                for feature in model_features:
                    if feature in medians:
                        try:
                            median_value = (
                                medians[
                                    feature
                                ]
                            )
                            df[
                                feature
                            ] = (
                                df[
                                    feature
                                ]
                                .fillna(
                                    float(
                                        median_value
                                    )
                                )
                            )
                        except Exception:
                            pass
            # ------------------------------------------------
            # Correct order
            # ------------------------------------------------
            df = df[
                model_features
            ]
            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------
            prediction = (
                model.predict(df)
            )
            predicted_value = int(
                np.asarray(
                    prediction
                )[0]
            )
            # ------------------------------------------------
            # Probability
            # ------------------------------------------------
            attack_probability = None
            if hasattr(
                model,
                "predict_proba"
            ):
                probabilities = (
                    model.predict_proba(
                        df
                    )
                )
                if (
                    probabilities.ndim == 2
                    and
                    probabilities.shape[1]
                    >= 2
                ):
                    attack_probability = (
                        float(
                            probabilities[
                                0
                            ][1]
                        )
                    )
            # ------------------------------------------------
            # Label
            # ------------------------------------------------
            if predicted_value == 1:
                label = "ATTACK"
                with self.lock:
                    self.attack_count += 1
            else:
                label = "NORMAL"
                with self.lock:
                    self.normal_count += 1
            with self.lock:
                self.flow_count += 1
                result = {
                    "Time":
                        datetime.now()
                        .strftime(
                            "%H:%M:%S"
                        ),
                    "Source":
                        (
                            f"{window.src_ip}:"
                            f"{window.src_port}"
                        ),
                    "Destination":
                        (
                            f"{window.dst_ip}:"
                            f"{window.dst_port}"
                        ),
                    "Protocol":
                        window.protocol,
                    "Packets":
                        window.packet_count,
                    "Packet Size":
                        round(
                            live_features[
                                "Tot size"
                            ],
                            2,
                        ),
                    "Rate":
                        round(
                            live_features[
                                "Rate"
                            ],
                            2,
                        ),
                    "Result":
                        label,
                    "Attack Probability":
                        (
                            round(
                                attack_probability
                                * 100,
                                2,
                            )
                            if attack_probability
                            is not None
                            else None
                        ),
                }
                self.results.appendleft(
                    result
                )
        except Exception as exc:
            self.error = (
                "Model prediction error: "
                + str(exc)
            )
    # --------------------------------------------------------
    # Capture loop
    # --------------------------------------------------------
    def capture_loop(self):
        try:
            while self.running:
                sniff(
                    iface=self.interface,
                    prn=self.process_packet,
                    store=False,
                    timeout=1,
                )
        except Exception as exc:
            self.error = (
                "Packet capture error: "
                + str(exc)
            )
        finally:
            self.running = False
    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------
    def start(
        self,
        interface
    ):
        if self.running:
            return
        self.interface = (
            interface
        )
        self.error = None
        self.running = True
        self.started_at = (
            datetime.now()
        )
        self.capture_thread = (
            threading.Thread(
                target=self.capture_loop,
                daemon=True,
            )
        )
        self.capture_thread.start()
    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------
    def stop(self):
        self.running = False

        # Discard incomplete packet windows.
        # The model is trained on complete 100-packet windows.
        with self.lock:
            self.current_windows.clear()
    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------
    def reset(self):
        with self.lock:
            self.results.clear()
            self.current_windows.clear()
            self.packet_count = 0
            self.flow_count = 0
            self.normal_count = 0
            self.attack_count = 0
        self.error = None
# ============================================================
# SESSION STATE
# ============================================================
if (
    "capture_manager"
    not in st.session_state
):
    st.session_state.capture_manager = (
        CaptureManager()
    )
manager = (
    st.session_state.capture_manager
)
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header(
        "🔴 Live Capture"
    )
    st.caption(
        "Only real network traffic is captured."
    )
    st.divider()
    interfaces = (
        get_friendly_interfaces()
    )
    if not interfaces:
        st.error(
            "No network interfaces detected."
        )
        st.info(
            "Make sure Npcap is installed."
        )
        st.stop()
    interface_names = list(
        interfaces.keys()
    )
    connected = [
        x
        for x in interface_names
        if "Connected" in x
    ]
    disconnected = [
        x
        for x in interface_names
        if "Disconnected" in x
    ]
    loopback = [
        x
        for x in interface_names
        if "Loopback" in x
    ]
    ordered_interfaces = (
        connected
        + disconnected
        + loopback
    )
    selected_name = st.selectbox(
        "Network interface",
        ordered_interfaces,
    )
    selected_interface = (
        interfaces[
            selected_name
        ]
    )
    st.divider()
    if not manager.running:
        if st.button(
            "▶ START MONITORING",
            type="primary",
            use_container_width=True,
        ):
            manager.reset()
            manager.start(
                selected_interface
            )
            st.rerun()
    else:
        if st.button(
            "⏹ STOP MONITORING",
            use_container_width=True,
        ):
            manager.stop()
            st.rerun()
    st.divider()
    if st.button(
        "🔄 Reset Results",
        use_container_width=True,
    ):
        manager.reset()
        st.rerun()
    st.divider()
    st.markdown(
        """
        **Live monitoring**
        • Real packets only
        • No synthetic traffic
        • No training CSVs
        • Existing trained model
        • 10-packet live windows
        """
    )
# ============================================================
# MAIN TITLE
# ============================================================
st.markdown(
    """
    <div class="main-title">
        🛡️ IoT Intrusion Detection System
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="sub-title">
        Live Network Traffic Monitor
    </div>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# LIVE DASHBOARD FRAGMENT
#
# IMPORTANT:
# ALL METRICS ARE INSIDE THE FRAGMENT.
# Therefore they update together with the results table.
# ============================================================
@st.fragment(
    run_every=REFRESH_SECONDS
)
def live_dashboard():
    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------
    if manager.running:
        st.success(
            f"🟢 MONITORING — {selected_name}"
        )
    else:
        st.warning(
            "⚪ STOPPED"
        )
    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    col1, col2, col3, col4 = (
        st.columns(4)
    )
    with manager.lock:
        packets = (
            manager.packet_count
        )
        analyzed = (
            manager.flow_count
        )
        normal = (
            manager.normal_count
        )
        attack = (
            manager.attack_count
        )
        active = len(
            manager.current_windows
        )
        results = list(
            manager.results
        )
        error = manager.error
    with col1:
        st.metric(
            "Packets Captured",
            packets,
        )
    with col2:
        st.metric(
            "Flows Analyzed",
            analyzed,
        )
    with col3:
        st.metric(
            "NORMAL",
            normal,
        )
    with col4:
        st.metric(
            "ATTACK",
            attack,
        )
    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------
    if error:
        st.error(
            error
        )
    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    st.subheader(
        "Live Detection Results"
    )
    st.caption(
        "Active packet windows: "
        f"{active}"
    )
    if not results:
        if manager.running:
            st.info(
                "Monitoring real traffic... "
                "Completed 100-packet windows "
                "will appear here."
            )
        else:
            st.info(
                "No traffic has been analyzed yet. "
                "Select a network interface and "
                "press START MONITORING."
            )
    else:
        result_df = pd.DataFrame(
            results
        )
        # ----------------------------------------------------
        # Format probability
        # ----------------------------------------------------
        if (
            "Attack Probability"
            in result_df.columns
        ):
            result_df[
                "Attack Probability"
            ] = result_df[
                "Attack Probability"
            ].apply(
                lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
            )
        # ----------------------------------------------------
        # Highlight result
        # ----------------------------------------------------
        def highlight_result(row):
            if (
                row["Result"]
                == "ATTACK"
            ):
                return [
                    "background-color: #5b1f1f"
                ] * len(row)
            return [
                "background-color: #163d2a"
            ] * len(row)
        styled_df = (
            result_df
            .style
            .apply(
                highlight_result,
                axis=1,
            )
        )
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
        )
    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------
    st.divider()
    st.caption(
        "Pipeline: real network packets → "
        "100-packet windows → selected CICIoT2023 "
        "features → trained Random Forest → "
        "NORMAL / ATTACK"
    )
    st.caption(
        "No training CSV or generated demo traffic "
        "is used during live monitoring."
    )
# ============================================================
# RUN LIVE DASHBOARD
# ============================================================
live_dashboard()
