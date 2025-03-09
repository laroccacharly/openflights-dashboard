export IMAGE_NAME="openflights-dashboard"
export CONTAINER_NAME="openflights-dashboard-container"
# Default to amd64 architecture
export DOCKER_DEFAULT_PLATFORM=linux/amd64
alias build="docker build -t $IMAGE_NAME ."
alias run="echo "http://localhost:8501" && docker run --name $CONTAINER_NAME -p 8501:8501 -v $(pwd)/src:/app/src $IMAGE_NAME"
alias start="echo "http://localhost:8501" && docker start -a $CONTAINER_NAME"
alias stop="docker stop $CONTAINER_NAME"
alias reload="docker restart $CONTAINER_NAME"
alias launch="fly launch --name openflights-dashboard --no-deploy"
alias deploy="fly deploy"