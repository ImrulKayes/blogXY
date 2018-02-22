import urllib2
import json
import time

def get_friends(user, logger):
    """Returns list of friends of a blogger"""

    friends = []
    page_number = 1

    # send the request to blogster
    try:
        result = json.load(urllib2.urlopen('http://www.blogster.com/a/friends.api/'+str(int(time.time()))+'?of='+user+'&num=100&page='+str(page_number)))
    except Exception as e:
        logger.error(e.message)
        return []

    # run as long as we are getting new friends
    while result['friends']:
        for friend in result['friends']:
            friends.append(friend['username'])
        page_number += 1
        try:    
            result = json.load(urllib2.urlopen('http://www.blogster.com/a/friends.api/1372369841842?of='+user+'&num=100&page='+str(page_number)))
        except Exception as e:
            logger.error(e.message)
            return []
    return friends                    
