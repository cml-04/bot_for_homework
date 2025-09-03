# Environment Setup Guide

This guide provides detailed instructions for setting up the distributed genetic algorithm framework on both Windows (head node) and Linux (worker nodes) systems.

## Prerequisites

- Python 3.8 or higher
- Network connectivity between head and worker nodes
- Administrative access for firewall configuration

## Windows Head Node Setup

### 1. Python Environment

```powershell
# Check Python version
python --version

# Create virtual environment
python -m venv distributed_ga_env

# Activate virtual environment
distributed_ga_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```powershell
# Clone repository
git clone <repository-url>
cd distributed_ga

# Install requirements
pip install -r requirements.txt

# Verify Ray installation
python -c "import ray; print(ray.__version__)"
```

### 3. Windows Firewall Configuration

**Option A: Using Windows Firewall with Advanced Security**

1. Open "Windows Defender Firewall with Advanced Security"
2. Click "Inbound Rules" → "New Rule"
3. Select "Port" → "TCP" → Specific ports: `10001, 8265`
4. Allow the connection
5. Apply to all profiles
6. Name the rule "Ray Distributed GA"

**Option B: Using Command Line**

```powershell
# Run as Administrator
netsh advfirewall firewall add rule name="Ray Head Node" dir=in action=allow protocol=TCP localport=10001
netsh advfirewall firewall add rule name="Ray Dashboard" dir=in action=allow protocol=TCP localport=8265
```

**Option C: Using PowerShell**

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "Ray Head Node" -Direction Inbound -Protocol TCP -LocalPort 10001 -Action Allow
New-NetFirewallRule -DisplayName "Ray Dashboard" -Direction Inbound -Protocol TCP -LocalPort 8265 -Action Allow
```

### 4. Network Configuration

Find your Windows machine's IP address:

```powershell
ipconfig /all
```

Note the IPv4 address for your network interface.

### 5. Test Head Node

```powershell
# Start head node
python scripts/start_head.py

# You should see output like:
# Ray head node started successfully!
# Dashboard URL: http://0.0.0.0:8265
# Workers can connect using: ray start --address='<IP>:10001'
```

## Linux Worker Node Setup

### 1. Python Environment

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv distributed_ga_env

# Activate virtual environment
source distributed_ga_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Clone repository (or copy from head node)
git clone <repository-url>
cd distributed_ga

# Install requirements
pip install -r requirements.txt

# Install additional system dependencies
sudo apt install gcc g++ -y

# Verify Ray installation
python -c "import ray; print(ray.__version__)"
```

### 3. Linux Firewall Configuration

**For Ubuntu/Debian (UFW):**

```bash
# Enable UFW if not already enabled
sudo ufw enable

# Allow incoming connections (if needed)
sudo ufw allow from <HEAD_NODE_IP> to any port 22

# Check status
sudo ufw status
```

**For CentOS/RHEL (firewalld):**

```bash
# Check firewall status
sudo firewall-cmd --state

# Allow SSH from head node (if needed)
sudo firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='<HEAD_NODE_IP>' service name='ssh' accept"

# Reload firewall
sudo firewall-cmd --reload
```

### 4. SSH Configuration (Optional)

If using SSH for remote management:

```bash
# Generate SSH key pair on head node
ssh-keygen -t rsa -b 4096 -C "distributed_ga@headnode"

# Copy public key to worker nodes
ssh-copy-id user@worker_node_ip

# Test SSH connection
ssh user@worker_node_ip
```

### 5. Test Worker Node

```bash
# Test connection to head node
ping <HEAD_NODE_IP>

# Start worker node
python scripts/start_worker.py --address ray://<HEAD_NODE_IP>:10001

# You should see output like:
# Connecting to Ray head node at: ray://<HEAD_NODE_IP>:10001
# Ray worker connected successfully!
```

## Docker Setup (Alternative)

### 1. Install Docker

**Windows:**
1. Download Docker Desktop from docker.com
2. Install and restart
3. Verify: `docker --version`

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Log out and back in for group changes
```

### 2. Build Images

```bash
# Build head node image
docker build -f docker/Dockerfile.head -t distributed_ga:head .

# Build worker node image
docker build -f docker/Dockerfile.worker -t distributed_ga:worker .
```

### 3. Run with Docker

**Head Node:**
```bash
docker run -d \
  --name ga_head \
  -p 10001:10001 \
  -p 8265:8265 \
  -v ${PWD}/results:/app/results \
  distributed_ga:head
```

**Worker Nodes:**
```bash
docker run -d \
  --name ga_worker_1 \
  -e RAY_HEAD_ADDRESS=ray://<HEAD_NODE_IP>:10001 \
  distributed_ga:worker
```

## Network Topology Examples

### Simple LAN Setup

```
Windows Laptop (Head Node)
IP: 192.168.1.100
Ports: 10001, 8265

Linux Server 1 (Worker)          Linux Server 2 (Worker)
IP: 192.168.1.101                IP: 192.168.1.102
Connects to: 192.168.1.100:10001  Connects to: 192.168.1.100:10001
```

### VPN/Cloud Setup

```
Windows Laptop (Head Node)
VPN IP: 10.0.1.100
Public IP: <external_ip>
Ports: 10001, 8265 (forwarded)

Cloud Server 1 (Worker)          Cloud Server 2 (Worker)
IP: 10.0.1.101                   IP: 10.0.1.102
Connects to: 10.0.1.100:10001    Connects to: 10.0.1.100:10001
```

### SSH Tunnel Setup

For restrictive network environments:

**On Worker Node:**
```bash
# Create SSH tunnel
ssh -L 10001:<HEAD_NODE_INTERNAL_IP>:10001 user@gateway_server

# In another terminal, connect worker
python scripts/start_worker.py --address ray://localhost:10001
```

## Performance Tuning

### System Configuration

**Linux Worker Nodes:**

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize network settings
echo "net.core.rmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Windows Head Node:**

```powershell
# Increase TCP connection limit (if needed)
# This typically requires registry modifications - consult system administrator
```

### Ray Configuration

**Head Node:**
```bash
# Start with specific resource allocation
python scripts/start_head.py \
  --object-store-memory 2000000000 \
  --host 0.0.0.0 \
  --port 10001
```

**Worker Nodes:**
```bash
# Start with specific resource limits
python scripts/start_worker.py \
  --address ray://<HEAD_NODE_IP>:10001 \
  --num-cpus 8 \
  --memory 16000000000 \
  --object-store-memory 1000000000
```

## Troubleshooting

### Connection Issues

1. **Port binding errors:**
   ```bash
   # Check if port is in use
   netstat -an | grep 10001
   
   # Kill process using port (Linux)
   sudo lsof -ti:10001 | xargs sudo kill -9
   ```

2. **Firewall blocking:**
   ```bash
   # Test port connectivity
   telnet <HEAD_NODE_IP> 10001
   
   # Or using nc
   nc -zv <HEAD_NODE_IP> 10001
   ```

3. **DNS resolution:**
   ```bash
   # Use IP addresses instead of hostnames
   nslookup <hostname>
   ```

### Performance Issues

1. **Memory problems:**
   ```bash
   # Check memory usage
   htop
   
   # Monitor Ray memory
   ray status
   ```

2. **Network bandwidth:**
   ```bash
   # Test network speed between nodes
   iperf3 -s  # On head node
   iperf3 -c <HEAD_NODE_IP>  # On worker
   ```

### Debugging

Enable verbose logging:
```bash
# Head node
python scripts/start_head.py --log-level DEBUG --verbose

# Worker node
python scripts/start_worker.py --address ray://<HEAD_NODE_IP>:10001 --log-level DEBUG --verbose
```

## Security Considerations

### Network Security

1. **Use VPN:** Set up VPN between head and worker nodes
2. **Firewall rules:** Only allow necessary ports
3. **SSH keys:** Use key-based authentication instead of passwords
4. **Network segmentation:** Isolate computational network

### Ray Security

```yaml
# In cluster_config.yaml
security:
  authentication:
    enabled: true
    token: "your-secure-token"
  encryption:
    enabled: true
```

### Best Practices

1. **Regular updates:** Keep Python and dependencies updated
2. **Monitoring:** Set up log monitoring and alerting
3. **Backup:** Regular backup of configurations and results
4. **Access control:** Limit access to computational resources

## Validation

### Test Basic Setup

```bash
# Test head node
python -c "
import ray
ray.init(address=None, port=10001)
print('Head node working')
ray.shutdown()
"

# Test worker connection
python -c "
import ray
ray.init(address='ray://<HEAD_NODE_IP>:10001')
print('Worker connection working')
ray.shutdown()
"
```

### Test Distributed GA

```bash
# Run simple test experiment
python scripts/run_experiment.py \
  --function sphere \
  --dimensions 2 \
  --population-size 20 \
  --max-generations 10 \
  --save-plots
```

## Next Steps

After successful setup:

1. **Configure optimization parameters** in `config/ga_config.yaml`
2. **Set up cluster configuration** in `config/cluster_config.yaml`
3. **Run performance benchmarks** to validate speedup
4. **Monitor system resources** during optimization
5. **Scale up** with additional worker nodes as needed

For production deployments, consider using container orchestration (Kubernetes) or cloud platforms (AWS, GCP, Azure) for better scalability and management.