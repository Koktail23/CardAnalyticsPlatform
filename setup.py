#!/usr/bin/env python
"""
Setup script for Card Analytics Platform
Initializes the environment and validates configuration
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple
import click
from rich.console import Console
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv

console = Console()


def check_python_version():
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        console.print("[red]Error: Python 3.11+ is required[/red]")
        console.print(f"Current version: {sys.version}")
        return False
    console.print(f"[green]✓[/green] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'],
                                capture_output=True, text=True, check=True)
        console.print(f"[green]✓[/green] Docker: {result.stdout.strip()}")

        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'ps'],
                                capture_output=True, text=True, check=True)
        console.print("[green]✓[/green] Docker daemon is running")
        return True
    except subprocess.CalledProcessError:
        console.print("[red]✗[/red] Docker is not installed or not running")
        return False
    except FileNotFoundError:
        console.print("[red]✗[/red] Docker command not found")
        return False


def check_docker_compose():
    """Check if Docker Compose is installed"""
    try:
        # Try docker compose (v2)
        result = subprocess.run(['docker', 'compose', 'version'],
                                capture_output=True, text=True, check=True)
        console.print(f"[green]✓[/green] Docker Compose: {result.stdout.strip()}")
        return True, 'docker compose'
    except:
        try:
            # Try docker-compose (v1)
            result = subprocess.run(['docker-compose', '--version'],
                                    capture_output=True, text=True, check=True)
            console.print(f"[green]✓[/green] Docker Compose: {result.stdout.strip()}")
            return True, 'docker-compose'
        except:
            console.print("[red]✗[/red] Docker Compose not found")
            return False, None


def create_directories():
    """Create required project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/clickhouse',
        'data/samples',
        'models/registry',
        'reports/notebooks',
        'reports/dashboards',
        'logs',
        'backups',
        'tests/unit',
        'tests/integration',
        'ml/models',
        'ml/features',
        'validation/expectations',
        'validation/reports'
    ]

    console.print("\n[bold]Creating project directories...[/bold]")
    for directory in track(directories, description="Creating directories"):
        Path(directory).mkdir(parents=True, exist_ok=True)

    console.print("[green]✓[/green] All directories created")


def setup_env_file():
    """Create .env file from .env.example"""
    if Path('.env').exists():
        console.print("[yellow]![/yellow] .env file already exists")
        if not click.confirm("Do you want to overwrite it?"):
            return True

    if Path('.env.example').exists():
        shutil.copy('.env.example', '.env')
        console.print("[green]✓[/green] Created .env file from .env.example")
        console.print("[yellow]![/yellow] Please update .env with your actual credentials")
        return True
    else:
        console.print("[red]✗[/red] .env.example not found")
        return False


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/raw/*
data/processed/*
data/clickhouse/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/samples/

# Models
models/registry/*
!models/registry/.gitkeep
*.pkl
*.joblib
*.h5

# Logs
logs/
*.log

# Environment
.env
.env.local
.env.*.local

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml

# Backups
backups/

# Reports
reports/notebooks/*.html
reports/dashboards/*.html

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover

# MCP
mcp_agents/__pycache__/
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())

    console.print("[green]✓[/green] Created .gitignore file")


def install_python_packages():
    """Install Python packages"""
    console.print("\n[bold]Installing Python packages...[/bold]")

    if not Path('requirements.txt').exists():
        console.print("[red]✗[/red] requirements.txt not found")
        return False

    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                       check=True, capture_output=True)
        console.print("[green]✓[/green] Updated pip")

        with console.status("[bold green]Installing packages...") as status:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                console.print("[green]✓[/green] All packages installed successfully")
                return True
            else:
                console.print("[red]✗[/red] Some packages failed to install")
                console.print(result.stderr)
                return False

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] Failed to install packages: {e}")
        return False


def start_docker_services():
    """Start Docker services"""
    console.print("\n[bold]Starting Docker services...[/bold]")

    _, compose_cmd = check_docker_compose()
    if not compose_cmd:
        return False

    try:
        # Build images
        with console.status("[bold green]Building Docker images...") as status:
            subprocess.run(f'{compose_cmd} build'.split(), check=True,
                           capture_output=True)
        console.print("[green]✓[/green] Docker images built")

        # Start services
        with console.status("[bold green]Starting services...") as status:
            subprocess.run(f'{compose_cmd} up -d'.split(), check=True,
                           capture_output=True)
        console.print("[green]✓[/green] Services started")

        # Show service status
        result = subprocess.run(f'{compose_cmd} ps'.split(),
                                capture_output=True, text=True)
        console.print("\n[bold]Service Status:[/bold]")
        console.print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] Failed to start services: {e}")
        return False


def test_clickhouse_connection():
    """Test ClickHouse connection"""
    console.print("\n[bold]Testing ClickHouse connection...[/bold]")

    try:
        import time
        time.sleep(5)  # Wait for ClickHouse to start

        from database.clickhouse_client import get_clickhouse_manager

        manager = get_clickhouse_manager()
        if manager.test_connection():
            console.print("[green]✓[/green] ClickHouse connection successful")

            # Get database info
            result = manager.query("SELECT version()")
            console.print(f"  ClickHouse version: {result[0][0]}")

            # Check tables
            tables = manager.query("""
                SELECT name 
                FROM system.tables 
                WHERE database = 'card_analytics'
                ORDER BY name
            """)

            if tables:
                console.print(f"  Found {len(tables)} tables in card_analytics database")
                table = Table(title="Database Tables")
                table.add_column("Table Name", style="cyan")
                for t in tables:
                    table.add_row(t[0])
                console.print(table)

            return True
        else:
            console.print("[red]✗[/red] ClickHouse connection failed")
            return False

    except Exception as e:
        console.print(f"[red]✗[/red] Error testing connection: {e}")
        return False


def generate_sample_data():
    """Generate sample data for testing"""
    console.print("\n[bold]Generating sample data...[/bold]")

    try:
        from etl.sample_data_generator import generate_sample_transactions

        # Generate sample data
        df = generate_sample_transactions(num_records=10000)

        # Save to CSV
        output_path = Path('data/samples/sample_transactions.csv')
        df.to_csv(output_path, index=False)

        console.print(f"[green]✓[/green] Generated {len(df)} sample transactions")
        console.print(f"  Saved to: {output_path}")

        # Show sample
        console.print("\n[bold]Sample data preview:[/bold]")
        console.print(df.head().to_string())

        return True

    except ImportError:
        console.print("[yellow]![/yellow] Sample data generator not found, skipping")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Error generating sample data: {e}")
        return False


@click.command()
@click.option('--skip-docker', is_flag=True, help='Skip Docker setup')
@click.option('--skip-packages', is_flag=True, help='Skip Python package installation')
@click.option('--generate-data', is_flag=True, help='Generate sample data')
def main(skip_docker, skip_packages, generate_data):
    """Setup Card Analytics Platform"""

    console.print("[bold blue]Card Analytics Platform Setup[/bold blue]\n")

    # Check prerequisites
    console.print("[bold]Checking prerequisites...[/bold]")

    if not check_python_version():
        sys.exit(1)

    if not skip_docker:
        if not check_docker():
            console.print("[yellow]![/yellow] Docker is required. Please install Docker first.")
            if not click.confirm("Continue without Docker?"):
                sys.exit(1)
            skip_docker = True

    # Create project structure
    create_directories()
    create_gitignore()

    # Setup environment
    if not setup_env_file():
        console.print("[red]Error setting up environment file[/red]")
        sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Install packages
    if not skip_packages:
        if not install_python_packages():
            console.print("[yellow]![/yellow] Some packages failed to install")
            if not click.confirm("Continue anyway?"):
                sys.exit(1)

    # Start Docker services
    if not skip_docker:
        if not start_docker_services():
            console.print("[yellow]![/yellow] Failed to start Docker services")
            if not click.confirm("Continue anyway?"):
                sys.exit(1)

        # Test ClickHouse connection
        test_clickhouse_connection()

    # Generate sample data if requested
    if generate_data:
        generate_sample_data()

    # Summary
    console.print("\n[bold green]Setup completed successfully![/bold green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Update .env file with your actual credentials")
    console.print("2. Run migrations: python database/migrations/__init__.py")
    console.print("3. Load your data: python etl/loader.py --file your_data.csv")
    console.print("4. Open Jupyter Lab: http://localhost:8888")
    console.print("5. Access ClickHouse: http://localhost:8123")

    console.print("\n[bold]Quick commands:[/bold]")
    console.print("  Start services:  docker compose up -d")
    console.print("  Stop services:   docker compose down")
    console.print("  View logs:       docker compose logs -f")
    console.print("  Run tests:       pytest tests/")
    console.print("  Generate data:   python etl/sample_data_generator.py")


if __name__ == '__main__':
    main()