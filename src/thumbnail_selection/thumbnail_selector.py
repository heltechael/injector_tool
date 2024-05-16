from src.thumbnail_selection.thumbnail_assessor import ThumbnailAssessor

class ThumbnailSelector:
    def __init__(self, config, thumbnails_by_class, load_thumbnail):
        self.config = config
        self.thumbnails_by_class = thumbnails_by_class
        self.load_thumbnail = load_thumbnail
        self.thumbnail_assessment_threshold = config.get('THUMBNAIL_ASSESSMENT_THRESHOLD')
        self.USE_BEST_THUMBNAILS = config.get('USE_BEST_THUMBNAILS')

    def select_thumbnails(self, classes):
        selected_thumbnails = {}
        for eppo_class in classes:
            if eppo_class in self.thumbnails_by_class:
                class_thumbnails = self.thumbnails_by_class[eppo_class]
                
                if self.USE_BEST_THUMBNAILS:
                    assessed_thumbnails = [(thumbnail, self.assess_thumbnail(thumbnail)) for thumbnail in class_thumbnails]
                    filtered_thumbnails = [thumbnail for thumbnail, score in assessed_thumbnails if score <= self.thumbnail_assessment_threshold]
                    sorted_thumbnails = sorted(filtered_thumbnails, key=lambda thumbnail: self.assess_thumbnail(thumbnail))
                    selected_thumbnails[eppo_class] = sorted_thumbnails
                else:
                    selected_thumbnails[eppo_class] = class_thumbnails

        return selected_thumbnails

    def assess_thumbnail(self, thumbnail):
        thumbnail_image = self.load_thumbnail(thumbnail)
        assessor = ThumbnailAssessor(self.config, "models/thumbnail_classifier_best.pth")
        assessment_score = assessor.assess(thumbnail_image)
        print(f"Assessment_score: {assessment_score}")
        return assessment_score