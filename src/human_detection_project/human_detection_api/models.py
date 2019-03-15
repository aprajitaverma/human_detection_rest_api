from django.db import models

# Create your models here.


class ImageClass(models.Model):
    """To save images from the detected feed."""

    im_title = models.CharField(max_length=200)
    im_path = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def set_image_path(self, path):
        self.im_path = path

    def __str__(self):
        return '%s %s' % (self.im_title, self.im_path)

