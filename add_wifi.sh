#!/bin/bash

# Script to add WiFi credentials to Ubuntu Server on Raspberry Pi
# Usage: ./add_wifi.sh "SSID" "PASSWORD"

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run with sudo privileges"
        exit 1
    fi
}

# Function to validate inputs
validate_inputs() {
    if [[ -z "$SSID" ]]; then
        print_error "SSID cannot be empty"
        exit 1
    fi
    
    if [[ -z "$PASSWORD" ]]; then
        print_error "Password cannot be empty"
        exit 1
    fi
    
    if [[ ${#PASSWORD} -lt 8 ]]; then
        print_warning "Password is less than 8 characters. Some networks may reject it."
    fi
}

# Function to check if NetworkManager is available
check_networkmanager() {
    if ! command -v nmcli &> /dev/null; then
        print_error "NetworkManager (nmcli) is not installed"
        print_info "Installing NetworkManager..."
        apt-get update
        apt-get install -y network-manager
    fi
    
    # Check if NetworkManager service is running
    if ! systemctl is-active --quiet NetworkManager; then
        print_info "Starting NetworkManager service..."
        systemctl start NetworkManager
    fi
}

# Function to add WiFi connection
add_wifi_connection() {
    print_info "Adding WiFi connection..."
    print_info "SSID: $SSID"
    
    # Check if connection already exists
    if nmcli connection show "$SSID" &> /dev/null; then
        print_warning "WiFi connection '$SSID' already exists. Removing old connection..."
        nmcli connection delete "$SSID"
    fi
    
    # Create a persistent WiFi connection profile
    print_info "Creating persistent connection profile..."
    nmcli connection add \
        type wifi \
        con-name "$SSID" \
        ifname "*" \
        ssid "$SSID" \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.psk "$PASSWORD" \
        ipv4.method auto \
        connection.autoconnect yes \
        connection.autoconnect-priority 0 || {
        print_error "Failed to create WiFi connection profile"
        exit 1
    }
    
    # Save connection to disk
    nmcli connection reload
    
    print_success "WiFi connection profile created and saved!"
}

# Function to verify connection
verify_connection() {
    # Check if the connection profile was saved
    if nmcli connection show "$SSID" &> /dev/null; then
        print_success "WiFi connection profile saved successfully!"
        print_info "The device will auto-connect when the '$SSID' network is available."
        return 0
    else
        print_error "Failed to find saved WiFi connection profile"
        return 1
    fi
}

# Main function
main() {
    print_info "WiFi Configuration Script for Ubuntu Server on Raspberry Pi"
    echo ""
    
    # Check if running as root
    check_root
    
    # Get SSID and PASSWORD from arguments or prompt
    if [[ $# -ge 2 ]]; then
        SSID="$1"
        PASSWORD="$2"
    else
        read -p "Enter WiFi SSID: " SSID
        read -s -p "Enter WiFi Password: " PASSWORD
        echo ""
    fi
    
    # Validate inputs
    validate_inputs
    
    # Check NetworkManager availability
    check_networkmanager
    
    # Add WiFi connection
    add_wifi_connection
    
    # Verify connection
    if verify_connection; then
        print_success "WiFi has been successfully configured and saved!"
        echo ""
        print_info "Your WiFi connection is now persistent and will reconnect automatically on reboot."
        exit 0
    else
        print_error "Failed to configure WiFi connection."
        exit 1
    fi
}

# Run main function
main "$@"
