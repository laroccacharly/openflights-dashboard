export IMAGE_NAME="openflights-dashboard"
export CONTAINER_NAME="openflights-dashboard-container"
# Default to amd64 architecture
export DOCKER_DEFAULT_PLATFORM=linux/amd64
alias build="docker build -t $IMAGE_NAME ."
alias run="docker run -p 8501:8501 -v $(pwd)/src:/app/src $IMAGE_NAME"
alias stop="docker stop $CONTAINER_NAME"
