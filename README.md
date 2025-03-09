# OpenFlights Dashboard

An interactive dashboard for visualizing and analyzing OpenFlights data, including airports and routes worldwide.


Link: 
https://openflights-dashboard.fly.dev
## Data Sources

The dashboard uses two main datasets from OpenFlights:

1. **Airports Database**: Contains information about airports worldwide, including location, codes, and other metadata.
2. **Routes Database**: Contains information about flight routes between airports, including airlines, equipment, and other details.


Link: [OpenFlights Data](https://openflights.org/data.php#:~:text=As%20of%20January%202017%2C%20the,entry%20contains%20the%20following%20information)

## Setup Instructions

### Prerequisites

- Docker installed on your system

### Getting Started

1. Clone this repository:
   ```
   git clone <repository-url>
   cd openflights-dashboard
   ```

2. Source the environment variables and aliases:
   ```
   source env.sh
   ```

## Usage

The project includes several helpful Docker commands defined as aliases in the `env.sh` file:

- `build`: Build the Docker image
  ```
  build
  ```

- `run`: Run the container with volume mounting for development
  ```
  run
  ```

- `start`: Start an existing container
  ```
  start
  ```

- `stop`: Stop the running container
  ```
  stop
  ```

## Accessing the Dashboard

Once running, the dashboard is accessible at:
- Local development: http://localhost:8501

## Deployment

The project is configured for deployment to [Fly.io](https://fly.io/):

- `launch`: Initialize a Fly.io application (without deploying)
  ```
  launch
  ```

- `deploy`: Deploy the application to Fly.io
  ```
  deploy
  ```

## Acknowledgements

- Data provided by [OpenFlights](https://openflights.org/)
