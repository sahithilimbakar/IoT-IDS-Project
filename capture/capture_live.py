from scapy.all import sniff, wrpcap, get_if_list
from pathlib import Path
from datetime import datetime
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent
BENIGN_DIR = PROJECT_DIR / "local_data" / "benign"

BENIGN_DIR.mkdir(parents=True, exist_ok=True)


def choose_interface():
    interfaces = get_if_list()

    print("\nAvailable network interfaces:\n")

    for i, iface in enumerate(interfaces):
        print(f"[{i}] {iface}")

    print()

    while True:
        try:
            choice = int(input("Enter the interface number to capture from: "))

            if 0 <= choice < len(interfaces):
                return interfaces[choice]

            print("Invalid number. Try again.")

        except ValueError:
            print("Please enter a number.")


def main():
    print("=" * 60)
    print("REAL NETWORK TRAFFIC CAPTURE")
    print("=" * 60)

    interface = choose_interface()

    print()
    print("Selected interface:")
    print(interface)

    print()
    print("CAPTURE TYPE: BENIGN / NORMAL TRAFFIC")
    print()
    print("During the capture:")
    print("- Use your computer normally.")
    print("- Browse websites normally.")
    print("- Use VS Code normally.")
    print("- Do not generate attack traffic.")
    print("- Do not run stress/flood tools.")
    print()

    input("Press ENTER to start the 5-minute capture...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = BENIGN_DIR / f"benign_{timestamp}.pcap"

    print()
    print("=" * 60)
    print("CAPTURE STARTED")
    print("=" * 60)
    print(f"Interface : {interface}")
    print("Duration  : 5 minutes")
    print(f"Output    : {output_file}")
    print()
    print("Use your computer normally now.")
    print()

    try:
        packets = sniff(
            iface=interface,
            timeout=300,
            store=True
        )

    except Exception as e:
        print()
        print("ERROR WHILE CAPTURING PACKETS:")
        print(e)
        print()
        print("Check that Npcap is installed and that the")
        print("selected interface is the correct network interface.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("CAPTURE FINISHED")
    print("=" * 60)

    print(f"Packets captured: {len(packets)}")

    if len(packets) == 0:
        print()
        print("No packets were captured.")
        print("Do not continue until this is fixed.")
        sys.exit(1)

    wrpcap(str(output_file), packets)

    print()
    print("REAL TRAFFIC SAVED SUCCESSFULLY")
    print(f"File: {output_file}")
    print()


if __name__ == "__main__":
    main()