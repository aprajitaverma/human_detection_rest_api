from rest_framework import serializers
from .models import ImageClass


class DetectionSerializer(serializers.Serializer):
    """Serializes the input"""

    checker = serializers.BooleanField()


class ImageSerializer(serializers.ModelSerializer):
    """Serializes the input for Image Response to the client."""

    class Meta:
        model = ImageClass()
        fields = ("im_title", "im_path", "created_at")
