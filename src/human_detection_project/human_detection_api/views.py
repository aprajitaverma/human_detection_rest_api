from rest_framework.response import Response
from rest_framework import generics
from rest_framework.views import APIView
from . import serializers
from rest_framework import status
import threading
from . import detection_files

on_off = "global"


class DetectAPI(APIView):
    """For detecting human"""

    def get(self, h):
        """Returns API view"""

        return Response({'API: Run human detection'})

    serializer_class = serializers.DetectionSerializer

    def post(self, request):
        """Start the script"""
        # url = 'http://192.168.10.73:8000/human_detection/detect/'
        serializer = serializers.DetectionSerializer(data=request.data)

        if serializer.is_valid():
            global on_off
            on_off = serializer.data.get('checker')
            print(on_off)
            if on_off is True:
                t1 = threading.Thread(target=detection_files.check_for_trespassers)
                t1.start()
                return Response({"message": "Running human detection", "status": "True"}, status=status.HTTP_200_OK)
            else:
                return Response({"message": "Stopped human detection", "status": "False"}, status=status.HTTP_200_OK)
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class SendDetectedImage(generics, APIView):
#     """Sends the detected images to the client."""
#
#     queryset = ImageClass.objects.all()
#     serializer_class = ImageSerializer



