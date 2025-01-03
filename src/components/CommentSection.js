import React, { useState, useEffect } from 'react';
import firebase from '../firebase';
import './CommentSection.css';

const CommentSection = ({ postId }) => {
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');

  useEffect(() => {
    const fetchComments = async () => {
      const commentsCollection = await firebase.firestore().collection('posts').doc(postId).collection('comments').get();
      const commentsData = commentsCollection.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      setComments(commentsData);
    };
    fetchComments();
  }, [postId]);

  const handleAddComment = async () => {
    if (newComment.trim() === '') return;
    const comment = {
      text: newComment,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
    };
    await firebase.firestore().collection('posts').doc(postId).collection('comments').add(comment);
    setComments([...comments, comment]);
    setNewComment('');
  };

  const handleEditComment = async (id, newText) => {
    await firebase.firestore().collection('posts').doc(postId).collection('comments').doc(id).update({ text: newText });
    setComments(comments.map(comment => (comment.id === id ? { ...comment, text: newText } : comment)));
  };

  const handleDeleteComment = async (id) => {
    await firebase.firestore().collection('posts').doc(postId).collection('comments').doc(id).delete();
    setComments(comments.filter(comment => comment.id !== id));
  };

  return (
    <div className="comment-section">
      <h3>Comments</h3>
      <div className="comments-list">
        {comments.map(comment => (
          <div key={comment.id} className="comment">
            <p>{comment.text}</p>
            <button onClick={() => handleEditComment(comment.id, prompt('Edit comment:', comment.text))}>Edit</button>
            <button onClick={() => handleDeleteComment(comment.id)}>Delete</button>
          </div>
        ))}
      </div>
      <div className="add-comment">
        <input
          type="text"
          placeholder="Add a comment"
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
        />
        <button onClick={handleAddComment}>Add Comment</button>
      </div>
    </div>
  );
};

export default CommentSection;
