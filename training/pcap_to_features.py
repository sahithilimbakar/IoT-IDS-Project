from scapy.all import PcapReader, IP, TCP, UDP
from pathlib import Path
import pandas as pd
import time


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PCAP_FILE = (
    PROJECT_DIR
    / "local_data"
    / "attack"
    / "attack_test.pcap"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "local_data"
    / "attack"
    / "attack_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Each feature row represents 100 real packets
WINDOW_SIZE = 100

# Process only the first 2.4 million REAL packets
MAX_PACKETS = 2_400_000

FEATURES = [
    "Number",
    "ack_flag_number",
    "HTTPS",
    "Tot size",
    "Rate",
]


# ============================================================
# CREATE FEATURES FOR ONE 100-PACKET WINDOW
# ============================================================

def create_features(packets):

    if not packets:
        return None

    packet_count = len(packets)

    ack_flags = []
    https_flags = []
    packet_sizes = []
    timestamps = []

    for packet in packets:

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        try:
            timestamps.append(float(packet.time))
        except Exception:
            pass

        # ----------------------------------------------------
        # Packet size
        # ----------------------------------------------------

        try:

            if IP in packet:
                packet_sizes.append(
                    int(packet[IP].len)
                )

            else:
                packet_sizes.append(
                    len(packet)
                )

        except Exception:

            packet_sizes.append(
                len(packet)
            )

        # ----------------------------------------------------
        # TCP
        # ----------------------------------------------------

        if TCP in packet:

            try:

                flags = int(
                    packet[TCP].flags
                )

                ack_flags.append(
                    1 if flags & 0x10 else 0
                )

            except Exception:

                ack_flags.append(0)

            try:

                sport = int(
                    packet[TCP].sport
                )

                dport = int(
                    packet[TCP].dport
                )

                https_flags.append(
                    1
                    if (
                        sport == 443
                        or dport == 443
                    )
                    else 0
                )

            except Exception:

                https_flags.append(0)

        # ----------------------------------------------------
        # UDP
        # ----------------------------------------------------

        elif UDP in packet:

            # UDP has no TCP ACK flag
            ack_flags.append(0)

            try:

                sport = int(
                    packet[UDP].sport
                )

                dport = int(
                    packet[UDP].dport
                )

                # UDP port 443 can represent QUIC/HTTPS traffic
                https_flags.append(
                    1
                    if (
                        sport == 443
                        or dport == 443
                    )
                    else 0
                )

            except Exception:

                https_flags.append(0)

        # ----------------------------------------------------
        # Other protocols
        # ----------------------------------------------------

        else:

            ack_flags.append(0)
            https_flags.append(0)

    # ========================================================
    # RATE
    # ========================================================

    if len(timestamps) >= 2:

        duration = (
            max(timestamps)
            - min(timestamps)
        )

        if duration > 0:

            rate = (
                packet_count
                / duration
            )

        else:

            rate = 0.0

    else:

        rate = 0.0

    # ========================================================
    # RETURN FIVE FEATURES
    # ========================================================

    return {

        "Number": packet_count,

        "ack_flag_number": (
            sum(ack_flags)
            / len(ack_flags)
            if ack_flags
            else 0.0
        ),

        "HTTPS": (
            sum(https_flags)
            / len(https_flags)
            if https_flags
            else 0.0
        ),

        "Tot size": (
            sum(packet_sizes)
            / len(packet_sizes)
            if packet_sizes
            else 0.0
        ),

        "Rate": rate,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("REAL ATTACK PCAP -> IDS FEATURES")
    print("=" * 60)

    print()
    print("Input PCAP:")
    print(PCAP_FILE)

    print()
    print("Output CSV:")
    print(OUTPUT_FILE)

    print()
    print("Window size:", WINDOW_SIZE)

    print(
        "Maximum packets:",
        f"{MAX_PACKETS:,}"
    )

    # ========================================================
    # CHECK PCAP
    # ========================================================

    if not PCAP_FILE.exists():

        print()
        print("ERROR: Attack PCAP was not found.")
        print(PCAP_FILE)

        return

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # VARIABLES
    # ========================================================

    rows = []

    packets = []

    packet_count = 0

    window_count = 0

    start_time = time.time()

    # ========================================================
    # STREAMING PCAP READER
    # ========================================================

    print()
    print("Reading real attack PCAP...")
    print(
        "Streaming mode enabled - "
        "the entire 2 GB PCAP is NOT loaded into RAM."
    )

    print()

    try:

        with PcapReader(
            str(PCAP_FILE)
        ) as reader:

            for packet in reader:

                # ------------------------------------------------
                # STOP AFTER 2.4 MILLION REAL PACKETS
                # ------------------------------------------------

                if packet_count >= MAX_PACKETS:

                    break

                # ------------------------------------------------
                # Store current packet
                # ------------------------------------------------

                packets.append(packet)

                packet_count += 1

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    packet_count
                    % 100000
                    == 0
                ):

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    print(
                        f"Packets processed: "
                        f"{packet_count:,} | "
                        f"Windows: "
                        f"{window_count:,} | "
                        f"Time: "
                        f"{elapsed:.1f}s",
                        flush=True
                    )

                # ------------------------------------------------
                # Create 100-packet window
                # ------------------------------------------------

                if len(packets) == WINDOW_SIZE:

                    features = create_features(
                        packets
                    )

                    if features is not None:

                        rows.append(
                            features
                        )

                        window_count += 1

                    # Clear current window
                    packets = []

    except KeyboardInterrupt:

        print()
        print(
            "Process interrupted by user."
        )

        print(
            f"Packets processed before stop: "
            f"{packet_count:,}"
        )

        return

    except Exception as e:

        print()
        print(
            "ERROR while reading PCAP:"
        )

        print(e)

        return

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 60)
    print("PCAP PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Total packets processed: "
        f"{packet_count:,}"
    )

    print(
        f"100-packet windows created: "
        f"{window_count:,}"
    )

    print(
        f"Processing time: "
        f"{elapsed / 60:.2f} minutes"
    )

    # ========================================================
    # CHECK FEATURES
    # ========================================================

    if not rows:

        print()
        print(
            "ERROR: No feature windows were created."
        )

        return

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows,
        columns=FEATURES
    )

    # ========================================================
    # CLEAN INVALID VALUES
    # ========================================================

    df = df.replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )

    df = df.dropna()

    # ========================================================
    # SAVE CSV
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Feature rows saved: "
        f"{len(df):,}"
    )

    print()
    print("Saved to:")

    print(OUTPUT_FILE)

    print()
    print("Features:")

    print(
        ", ".join(FEATURES)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()