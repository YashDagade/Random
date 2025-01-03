import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import firebase from '../firebase';
import './BlogPost.css';

const BlogPost = () => {
  const { id } = useParams();
  const [post, setPost] = useState(null);
  const [views, setViews] = useState(0);
  const [likes, setLikes] = useState(0);
  const [comments, setComments] = useState([]);

  useEffect(() => {
    const fetchPost = async () => {
      const postRef = firebase.firestore().collection('posts').doc(id);
      const doc = await postRef.get();
      if (doc.exists) {
        setPost(doc.data());
        setViews(doc.data().views || 0);
        setLikes(doc.data().likes || 0);
        setComments(doc.data().comments || []);
        postRef.update({ views: views + 1 });
      }
    };
    fetchPost();
  }, [id, views]);

  const handleLike = () => {
    const postRef = firebase.firestore().collection('posts').doc(id);
    postRef.update({ likes: likes + 1 });
    setLikes(likes + 1);
  };

  const handleComment = (comment) => {
    const postRef = firebase.firestore().collection('posts').doc(id);
    const newComments = [...comments, comment];
    postRef.update({ comments: newComments });
    setComments(newComments);
  };

  if (!post) {
    return <div>Loading...</div>;
  }

  return (
    <div className="blog-post">
      <h1>{post.title}</h1>
      <p>{post.content}</p>
      <div className="blog-post-footer">
        <span>Views: {views}</span>
        <button onClick={handleLike}>Like ({likes})</button>
        <div className="comments-section">
          <h3>Comments</h3>
          {comments.map((comment, index) => (
            <p key={index}>{comment}</p>
          ))}
          <input
            type="text"
            placeholder="Add a comment"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleComment(e.target.value);
                e.target.value = '';
              }
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default BlogPost;
